#include <dlfcn.h>

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "infer_engine.h"
#include "level_zero_min.h"

static bool checked_add(size_t left, size_t right, size_t *result) {
  if (right > SIZE_MAX - left) {
    return false;
  }
  *result = left + right;
  return true;
}

static bool checked_mul(size_t left, size_t right, size_t *result) {
  if (left != 0 && right > SIZE_MAX / left) {
    return false;
  }
  *result = left * right;
  return true;
}

#define INFER_AUTO_INDEX NPUNLOCK_INFER_AUTO_INDEX
#define INFER_PAGE_SIZE 4096u
#define INFER_FENCE_TIMEOUT_NS 5000000000ull

typedef npunlock_infer_request_input infer_request_input;
typedef npunlock_infer_request infer_request;
typedef npunlock_infer_result_output infer_result_output;
typedef npunlock_infer_result infer_result;

typedef struct graph_argument {
  uint32_t index;
  uint32_t type;
  uint32_t precision;
  uint32_t dims_count;
  uint32_t dims[5];
  uint8_t name[256];
  size_t name_size;
  size_t data_size;
  size_t request_input_index;
  void *allocation;
} graph_argument;

typedef struct ze_functions {
  npunlock_ze_init_fn init;
  npunlock_ze_driver_get_fn driver_get;
  npunlock_ze_init_drivers_fn init_drivers;
  npunlock_ze_driver_get_properties_fn driver_get_properties;
  npunlock_ze_driver_get_extensions_fn driver_get_extensions;
  npunlock_ze_driver_get_extension_address_fn driver_get_extension_address;
  npunlock_ze_device_get_fn device_get;
  npunlock_ze_device_get_properties_fn device_get_properties;
  npunlock_ze_context_create_fn context_create;
  npunlock_ze_context_destroy_fn context_destroy;
  npunlock_ze_device_get_queue_groups_fn device_get_queue_groups;
  npunlock_ze_mem_alloc_host_fn mem_alloc_host;
  npunlock_ze_mem_free_fn mem_free;
  npunlock_ze_command_queue_create_fn command_queue_create;
  npunlock_ze_command_queue_destroy_fn command_queue_destroy;
  npunlock_ze_command_queue_execute_fn command_queue_execute;
  npunlock_ze_command_list_create_fn command_list_create;
  npunlock_ze_command_list_close_fn command_list_close;
  npunlock_ze_command_list_destroy_fn command_list_destroy;
  npunlock_ze_fence_create_fn fence_create;
  npunlock_ze_fence_destroy_fn fence_destroy;
  npunlock_ze_fence_synchronize_fn fence_synchronize;
} ze_functions;

typedef struct graph_functions {
  npunlock_ze_graph_create2_fn create2;
  npunlock_ze_graph_destroy_fn destroy;
  npunlock_ze_graph_get_properties3_fn get_properties3;
  npunlock_ze_graph_get_argument_properties3_fn get_argument_properties3;
  npunlock_ze_graph_set_argument_fn set_argument;
  npunlock_ze_graph_initialize_fn initialize;
  npunlock_ze_graph_append_initialize_fn append_initialize;
  npunlock_ze_graph_append_execute_fn append_execute;
} graph_functions;

_Static_assert(offsetof(npunlock_ze_device_properties, type) == 16,
               "Level Zero device property ABI mismatch");
_Static_assert(sizeof(npunlock_ze_device_properties) == 368,
               "Level Zero device property ABI mismatch");
_Static_assert(sizeof(npunlock_ze_graph_desc_2) == 56, "Level Zero graph descriptor ABI mismatch");
_Static_assert(sizeof(npunlock_ze_graph_properties_3) == 32,
               "Level Zero graph properties ABI mismatch");
_Static_assert(offsetof(npunlock_ze_graph_argument_properties_3, dims_count) == 320,
               "Level Zero graph argument ABI mismatch");
_Static_assert(sizeof(npunlock_ze_graph_argument_properties_3) == 8776,
               "Level Zero graph argument ABI mismatch");
_Static_assert(sizeof(npunlock_ze_command_queue_group_properties) == 40,
               "Level Zero queue group ABI mismatch");
_Static_assert(sizeof(npunlock_ze_command_queue_desc) == 40,
               "Level Zero command queue ABI mismatch");
_Static_assert(sizeof(npunlock_ze_command_list_desc) == 24, "Level Zero command list ABI mismatch");
_Static_assert(sizeof(npunlock_ze_fence_desc) == 24, "Level Zero fence ABI mismatch");
_Static_assert(sizeof(npunlock_ze_host_mem_alloc_desc) == 24,
               "Level Zero host allocation ABI mismatch");

static void set_diagnostic(infer_result *result, const char *message) {
  size_t size;
  if (result->diagnostic != NULL || message == NULL) {
    return;
  }
  size = strlen(message);
  if (size > NPUNLOCK_INFER_MAX_DIAGNOSTIC_SIZE) {
    size = NPUNLOCK_INFER_MAX_DIAGNOSTIC_SIZE;
  }
  result->diagnostic = (char *)malloc(size + 1);
  if (result->diagnostic != NULL) {
    memcpy(result->diagnostic, message, size);
    result->diagnostic[size] = '\0';
  }
}

static void set_driver_error(infer_result *result, uint32_t code, const char *stage) {
  char message[192];
  result->driver_result = code;
  result->status = NPUNLOCK_INFER_DRIVER_FAILED;
  snprintf(message, sizeof(message), "%s returned Level Zero result 0x%08x", stage, code);
  set_diagnostic(result, message);
}

typedef void *infer_library;

static infer_library open_loader(void) {
  return dlopen("libze_loader.so.1", RTLD_NOW | RTLD_LOCAL);
}

static void close_loader(infer_library library) { dlclose(library); }

static bool load_export(infer_library library, const char *name, void *destination, size_t size) {
  void *proc = dlsym(library, name);
  if (proc == NULL || size != sizeof(proc)) {
    return false;
  }
  memcpy(destination, &proc, size);
  return true;
}

static bool load_ze_functions(infer_library library, ze_functions *ze) {
  bool required;
  memset(ze, 0, sizeof(*ze));
#define LOAD(name, field) load_export(library, name, &ze->field, sizeof(ze->field))
  required =
      LOAD("zeDriverGetProperties", driver_get_properties) &&
      LOAD("zeDriverGetExtensionProperties", driver_get_extensions) &&
      LOAD("zeDriverGetExtensionFunctionAddress", driver_get_extension_address) &&
      LOAD("zeDeviceGet", device_get) && LOAD("zeDeviceGetProperties", device_get_properties) &&
      LOAD("zeContextCreate", context_create) && LOAD("zeContextDestroy", context_destroy) &&
      LOAD("zeDeviceGetCommandQueueGroupProperties", device_get_queue_groups) &&
      LOAD("zeMemAllocHost", mem_alloc_host) && LOAD("zeMemFree", mem_free) &&
      LOAD("zeCommandQueueCreate", command_queue_create) &&
      LOAD("zeCommandQueueDestroy", command_queue_destroy) &&
      LOAD("zeCommandQueueExecuteCommandLists", command_queue_execute) &&
      LOAD("zeCommandListCreate", command_list_create) &&
      LOAD("zeCommandListClose", command_list_close) &&
      LOAD("zeCommandListDestroy", command_list_destroy) && LOAD("zeFenceCreate", fence_create) &&
      LOAD("zeFenceDestroy", fence_destroy) && LOAD("zeFenceHostSynchronize", fence_synchronize);
  if (!required) {
    return false;
  }
  if (LOAD("zeInitDrivers", init_drivers)) {
    return true;
  }
  return LOAD("zeInit", init) && LOAD("zeDriverGet", driver_get);
#undef LOAD
}

static bool select_npu(const infer_request *request, const ze_functions *ze, infer_result *result,
                       npunlock_ze_driver_handle *selected_driver,
                       npunlock_ze_device_handle *selected_device) {
  npunlock_ze_init_driver_type_desc init = {NPUNLOCK_ZE_STRUCTURE_TYPE_INIT_DRIVER_TYPE_DESC, NULL,
                                            NPUNLOCK_ZE_INIT_DRIVER_TYPE_FLAG_NPU};
  npunlock_ze_driver_handle *drivers = NULL;
  uint32_t driver_count = 0;
  uint32_t driver_index;
  uint32_t code;
  if (ze->init_drivers != NULL) {
    code = ze->init_drivers(&driver_count, NULL, &init);
  } else {
    code = ze->init(0);
    if (code == NPUNLOCK_ZE_SUCCESS) {
      code = ze->driver_get(&driver_count, NULL);
    }
  }
  if (code != NPUNLOCK_ZE_SUCCESS) {
    set_driver_error(result, code,
                     ze->init_drivers != NULL ? "zeInitDrivers(count)"
                                              : "zeInit/zeDriverGet(count)");
    return false;
  }
  if (driver_count == 0 ||
      (request->driver_index != INFER_AUTO_INDEX && request->driver_index >= driver_count)) {
    result->status = NPUNLOCK_INFER_NPU_NOT_FOUND;
    set_diagnostic(result, "requested NPU Level Zero driver was not found");
    return false;
  }
  drivers = (npunlock_ze_driver_handle *)calloc(driver_count, sizeof(*drivers));
  if (drivers == NULL) {
    result->status = NPUNLOCK_INFER_OUT_OF_MEMORY;
    return false;
  }
  code = ze->init_drivers != NULL ? ze->init_drivers(&driver_count, drivers, &init)
                                  : ze->driver_get(&driver_count, drivers);
  if (code != NPUNLOCK_ZE_SUCCESS) {
    set_driver_error(result, code,
                     ze->init_drivers != NULL ? "zeInitDrivers(list)" : "zeDriverGet(list)");
    free(drivers);
    return false;
  }
  for (driver_index = 0; driver_index < driver_count; ++driver_index) {
    npunlock_ze_device_handle *devices = NULL;
    uint32_t device_count = 0;
    uint32_t device_index;
    if (request->driver_index != INFER_AUTO_INDEX && request->driver_index != driver_index) {
      continue;
    }
    code = ze->device_get(drivers[driver_index], &device_count, NULL);
    if (code != NPUNLOCK_ZE_SUCCESS || device_count == 0) {
      continue;
    }
    devices = (npunlock_ze_device_handle *)calloc(device_count, sizeof(*devices));
    if (devices == NULL) {
      result->status = NPUNLOCK_INFER_OUT_OF_MEMORY;
      free(drivers);
      return false;
    }
    code = ze->device_get(drivers[driver_index], &device_count, devices);
    if (code != NPUNLOCK_ZE_SUCCESS) {
      free(devices);
      continue;
    }
    for (device_index = 0; device_index < device_count; ++device_index) {
      npunlock_ze_device_properties properties = {0};
      if (request->device_index != INFER_AUTO_INDEX && request->device_index != device_index) {
        continue;
      }
      properties.stype = NPUNLOCK_ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
      code = ze->device_get_properties(devices[device_index], &properties);
      if (code == NPUNLOCK_ZE_SUCCESS && properties.type == NPUNLOCK_ZE_DEVICE_TYPE_VPU) {
        *selected_driver = drivers[driver_index];
        *selected_device = devices[device_index];
        result->driver_index = driver_index;
        result->device_index = device_index;
        result->vendor_id = properties.vendorId;
        result->device_id = properties.deviceId;
        free(devices);
        free(drivers);
        return true;
      }
    }
    free(devices);
  }
  free(drivers);
  result->status = NPUNLOCK_INFER_NPU_NOT_FOUND;
  set_diagnostic(result, "requested Level Zero VPU device was not found");
  return false;
}

static uint32_t find_graph_version(const ze_functions *ze, npunlock_ze_driver_handle driver,
                                   infer_result *result) {
  npunlock_ze_driver_extension_properties *extensions = NULL;
  uint32_t count = 0;
  uint32_t index;
  uint32_t version = 0;
  uint32_t code = ze->driver_get_extensions(driver, &count, NULL);
  if (code != NPUNLOCK_ZE_SUCCESS) {
    set_driver_error(result, code, "zeDriverGetExtensionProperties(count)");
    return 0;
  }
  extensions = (npunlock_ze_driver_extension_properties *)calloc(count, sizeof(*extensions));
  if (extensions == NULL && count != 0) {
    result->status = NPUNLOCK_INFER_OUT_OF_MEMORY;
    return 0;
  }
  code = ze->driver_get_extensions(driver, &count, extensions);
  if (code != NPUNLOCK_ZE_SUCCESS) {
    set_driver_error(result, code, "zeDriverGetExtensionProperties(list)");
    free(extensions);
    return 0;
  }
  for (index = 0; index < count; ++index) {
    if (memchr(extensions[index].name, 0, sizeof(extensions[index].name)) != NULL &&
        strcmp(extensions[index].name, "ZE_extension_graph") == 0) {
      version = extensions[index].version;
      break;
    }
  }
  free(extensions);
  if (version == 0) {
    result->status = NPUNLOCK_INFER_GRAPH_EXTENSION_MISSING;
    set_diagnostic(result, "selected NPU driver does not advertise ZE_extension_graph");
  }
  return version;
}

static bool get_graph_functions(const ze_functions *ze, npunlock_ze_driver_handle driver,
                                uint32_t version, infer_result *result, graph_functions *graph) {
  void *raw_driver_ddi = NULL;
  npunlock_ze_npu_get_extension_fn get_extension = NULL;
  npunlock_ze_graph_ddi_prefix *ddi = NULL;
  npunlock_ze_driver_extension_npu request = {0};
  uint32_t code =
      ze->driver_get_extension_address(driver, "ZE_extension_driver_npu", &raw_driver_ddi);
  if (code != NPUNLOCK_ZE_SUCCESS || raw_driver_ddi == NULL) {
    set_driver_error(result, code, "zeDriverGetExtensionFunctionAddress");
    return false;
  }
  memcpy(&get_extension, raw_driver_ddi, sizeof(get_extension));
  if (get_extension == NULL) {
    result->status = NPUNLOCK_INFER_SYMBOL_MISSING;
    set_diagnostic(result, "NPU driver extension table is invalid");
    return false;
  }
  request.stype = NPUNLOCK_ZE_STRUCTURE_TYPE_DRIVER_EXTENSION_NPU_EXT;
  request.name = "ZE_extension_graph";
  request.version = version;
  request.ppFunctionAddress = (void **)&ddi;
  code = get_extension(driver, &request);
  if (code != NPUNLOCK_ZE_SUCCESS || ddi == NULL) {
    set_driver_error(result, code, "ZE_extension_driver_npu::pfnGetExtension");
    return false;
  }
  memset(graph, 0, sizeof(*graph));
#define GET(slot, field) memcpy(&graph->field, &ddi->slots[slot], sizeof(graph->field))
  GET(NPUNLOCK_ZE_GRAPH_DDI_CREATE2, create2);
  GET(NPUNLOCK_ZE_GRAPH_DDI_DESTROY, destroy);
  GET(NPUNLOCK_ZE_GRAPH_DDI_GET_PROPERTIES3, get_properties3);
  GET(NPUNLOCK_ZE_GRAPH_DDI_GET_ARGUMENT_PROPERTIES3, get_argument_properties3);
  GET(NPUNLOCK_ZE_GRAPH_DDI_SET_ARGUMENT, set_argument);
  GET(NPUNLOCK_ZE_GRAPH_DDI_INITIALIZE, initialize);
  GET(NPUNLOCK_ZE_GRAPH_DDI_APPEND_INITIALIZE, append_initialize);
  GET(NPUNLOCK_ZE_GRAPH_DDI_APPEND_EXECUTE, append_execute);
#undef GET
  if (graph->create2 == NULL || graph->destroy == NULL || graph->get_properties3 == NULL ||
      graph->get_argument_properties3 == NULL || graph->set_argument == NULL ||
      graph->append_execute == NULL) {
    result->status = NPUNLOCK_INFER_UNSUPPORTED;
    set_diagnostic(result, "required native graph execution entry point is unavailable");
    return false;
  }
  return true;
}

static bool argument_data_size(const npunlock_ze_graph_argument_properties_3 *properties,
                               size_t *data_size) {
  size_t elements = 1;
  size_t element_size;
  uint32_t index;
  if (properties->dims_count == 0 || properties->dims_count > 5) {
    return false;
  }
  if (properties->devicePrecision == NPUNLOCK_ZE_GRAPH_ARGUMENT_PRECISION_FP16) {
    element_size = sizeof(uint16_t);
  } else if (properties->devicePrecision == NPUNLOCK_ZE_GRAPH_ARGUMENT_PRECISION_FP32) {
    element_size = sizeof(uint32_t);
  } else {
    return false;
  }
  for (index = 0; index < properties->dims_count; ++index) {
    if (properties->dims[index] == 0 ||
        !checked_mul(elements, properties->dims[index], &elements)) {
      return false;
    }
  }
  return checked_mul(elements, element_size, data_size) && *data_size != 0 &&
         *data_size <= 64u * 1024u * 1024u;
}

static bool input_matches(const infer_request_input *input, const graph_argument *argument) {
  if (input->argument_index != INFER_AUTO_INDEX) {
    return input->argument_index == argument->index;
  }
  return input->name_size == argument->name_size &&
         memcmp(input->name, argument->name, input->name_size) == 0;
}

static bool query_arguments(const infer_request *request, const graph_functions *graph,
                            npunlock_ze_graph_handle handle, infer_result *result,
                            graph_argument **arguments_out, size_t *argument_count_out) {
  npunlock_ze_graph_properties_3 properties = {0};
  graph_argument *arguments = NULL;
  bool matched_inputs[NPUNLOCK_INFER_MAX_ARGUMENTS] = {false};
  size_t graph_input_count = 0;
  size_t output_count = 0;
  size_t total_output_size = 0;
  uint32_t index;
  uint32_t code;
  properties.stype = NPUNLOCK_ZE_STRUCTURE_TYPE_GRAPH_PROPERTIES_3;
  code = graph->get_properties3(handle, &properties);
  if (code != NPUNLOCK_ZE_SUCCESS) {
    set_driver_error(result, code, "pfnGetProperties3");
    return false;
  }
  if (properties.numGraphArgs == 0 || properties.numGraphArgs > NPUNLOCK_INFER_MAX_ARGUMENTS) {
    result->status = NPUNLOCK_INFER_UNSUPPORTED;
    set_diagnostic(result, "graph argument count is outside the supported bound");
    return false;
  }
  arguments = (graph_argument *)calloc(properties.numGraphArgs, sizeof(*arguments));
  if (arguments == NULL) {
    result->status = NPUNLOCK_INFER_OUT_OF_MEMORY;
    return false;
  }
  for (index = 0; index < properties.numGraphArgs; ++index) {
    npunlock_ze_graph_argument_properties_3 metadata = {0};
    graph_argument *argument = &arguments[index];
    const char *name_end;
    size_t input_index;
    metadata.stype = NPUNLOCK_ZE_STRUCTURE_TYPE_GRAPH_ARGUMENT_PROPERTIES_3;
    code = graph->get_argument_properties3(handle, index, &metadata);
    if (code != NPUNLOCK_ZE_SUCCESS) {
      set_driver_error(result, code, "pfnGetArgumentProperties3");
      free(arguments);
      return false;
    }
    if ((metadata.type != NPUNLOCK_ZE_GRAPH_ARGUMENT_TYPE_INPUT &&
         metadata.type != NPUNLOCK_ZE_GRAPH_ARGUMENT_TYPE_OUTPUT) ||
        !argument_data_size(&metadata, &argument->data_size)) {
      result->status = NPUNLOCK_INFER_UNSUPPORTED;
      set_diagnostic(result,
                     "graphinfer supports only bounded static FP16/FP32 inputs and outputs");
      free(arguments);
      return false;
    }
    name_end = (const char *)memchr(metadata.name, 0, sizeof(metadata.name));
    if (name_end == NULL) {
      result->status = NPUNLOCK_INFER_BAD_RESULT;
      set_diagnostic(result, "graph argument name is not terminated");
      free(arguments);
      return false;
    }
    argument->index = index;
    argument->type = metadata.type;
    argument->precision = metadata.devicePrecision;
    argument->dims_count = metadata.dims_count;
    memcpy(argument->dims, metadata.dims, sizeof(argument->dims));
    argument->name_size = (size_t)(name_end - metadata.name);
    memcpy(argument->name, metadata.name, argument->name_size);
    argument->request_input_index = SIZE_MAX;
    if (argument->type == NPUNLOCK_ZE_GRAPH_ARGUMENT_TYPE_INPUT) {
      ++graph_input_count;
      for (input_index = 0; input_index < request->input_count; ++input_index) {
        if (input_matches(&request->inputs[input_index], argument)) {
          if (argument->request_input_index != SIZE_MAX || matched_inputs[input_index]) {
            result->status = NPUNLOCK_INFER_BAD_REQUEST;
            set_diagnostic(result, "input selector is ambiguous or duplicated");
            free(arguments);
            return false;
          }
          argument->request_input_index = input_index;
          matched_inputs[input_index] = true;
        }
      }
      if (argument->request_input_index == SIZE_MAX ||
          request->inputs[argument->request_input_index].data_size != argument->data_size) {
        result->status = NPUNLOCK_INFER_BAD_REQUEST;
        set_diagnostic(result, "graph input is missing or has an incorrect byte size");
        free(arguments);
        return false;
      }
    } else {
      ++output_count;
      if (!checked_add(total_output_size, argument->name_size, &total_output_size) ||
          !checked_add(total_output_size, argument->data_size, &total_output_size) ||
          total_output_size > NPUNLOCK_INFER_MAX_OUTPUT_SIZE) {
        result->status = NPUNLOCK_INFER_UNSUPPORTED;
        set_diagnostic(result, "graph output payload exceeds the supported bound");
        free(arguments);
        return false;
      }
    }
  }
  if (graph_input_count != request->input_count || output_count == 0) {
    result->status = NPUNLOCK_INFER_BAD_REQUEST;
    set_diagnostic(result, "caller inputs do not exactly cover graph inputs");
    free(arguments);
    return false;
  }
  for (index = 0; index < request->input_count; ++index) {
    if (!matched_inputs[index]) {
      result->status = NPUNLOCK_INFER_BAD_REQUEST;
      set_diagnostic(result, "caller supplied an input selector not present in the graph");
      free(arguments);
      return false;
    }
  }
  *arguments_out = arguments;
  *argument_count_out = properties.numGraphArgs;
  return true;
}

static bool find_compute_queue(const ze_functions *ze, npunlock_ze_device_handle device,
                               infer_result *result, uint32_t *ordinal) {
  npunlock_ze_command_queue_group_properties *groups;
  uint32_t count = 0;
  uint32_t index;
  uint32_t code = ze->device_get_queue_groups(device, &count, NULL);
  if (code != NPUNLOCK_ZE_SUCCESS) {
    set_driver_error(result, code, "zeDeviceGetCommandQueueGroupProperties(count)");
    return false;
  }
  groups = (npunlock_ze_command_queue_group_properties *)calloc(count, sizeof(*groups));
  if (groups == NULL && count != 0) {
    result->status = NPUNLOCK_INFER_OUT_OF_MEMORY;
    return false;
  }
  for (index = 0; index < count; ++index) {
    groups[index].stype = NPUNLOCK_ZE_STRUCTURE_TYPE_COMMAND_QUEUE_GROUP_PROPERTIES;
  }
  code = ze->device_get_queue_groups(device, &count, groups);
  if (code != NPUNLOCK_ZE_SUCCESS) {
    set_driver_error(result, code, "zeDeviceGetCommandQueueGroupProperties(list)");
    free(groups);
    return false;
  }
  for (index = 0; index < count; ++index) {
    if (groups[index].numQueues != 0 &&
        (groups[index].flags & NPUNLOCK_ZE_COMMAND_QUEUE_GROUP_COMPUTE) != 0) {
      *ordinal = index;
      free(groups);
      return true;
    }
  }
  free(groups);
  result->status = NPUNLOCK_INFER_UNSUPPORTED;
  set_diagnostic(result, "selected NPU exposes no compute command queue group");
  return false;
}

static bool submit(const ze_functions *ze, const graph_functions *graph,
                   npunlock_ze_context_handle context, npunlock_ze_device_handle device,
                   npunlock_ze_graph_handle handle, uint32_t ordinal, bool initialize,
                   infer_result *result) {
  npunlock_ze_command_queue_desc queue_desc = {0};
  npunlock_ze_command_list_desc list_desc = {0};
  npunlock_ze_fence_desc fence_desc = {0};
  npunlock_ze_command_queue_handle queue = NULL;
  npunlock_ze_command_list_handle list = NULL;
  npunlock_ze_fence_handle fence = NULL;
  uint32_t code;
  bool success = false;
  queue_desc.stype = NPUNLOCK_ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC;
  queue_desc.ordinal = ordinal;
  list_desc.stype = NPUNLOCK_ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC;
  list_desc.commandQueueGroupOrdinal = ordinal;
  fence_desc.stype = NPUNLOCK_ZE_STRUCTURE_TYPE_FENCE_DESC;
  code = ze->command_queue_create(context, device, &queue_desc, &queue);
  if (code != NPUNLOCK_ZE_SUCCESS) {
    set_driver_error(result, code, "zeCommandQueueCreate");
    goto done;
  }
  code = ze->command_list_create(context, device, &list_desc, &list);
  if (code != NPUNLOCK_ZE_SUCCESS) {
    set_driver_error(result, code, "zeCommandListCreate");
    goto done;
  }
  code = ze->fence_create(queue, &fence_desc, &fence);
  if (code != NPUNLOCK_ZE_SUCCESS) {
    set_driver_error(result, code, "zeFenceCreate");
    goto done;
  }
  code = initialize ? graph->append_initialize(list, handle, NULL, 0, NULL)
                    : graph->append_execute(list, handle, NULL, NULL, 0, NULL);
  if (code != NPUNLOCK_ZE_SUCCESS) {
    set_driver_error(result, code,
                     initialize ? "pfnAppendGraphInitialize" : "pfnAppendGraphExecute");
    goto done;
  }
  code = ze->command_list_close(list);
  if (code != NPUNLOCK_ZE_SUCCESS) {
    set_driver_error(result, code, "zeCommandListClose");
    goto done;
  }
  code = ze->command_queue_execute(queue, 1, &list, fence);
  if (code != NPUNLOCK_ZE_SUCCESS) {
    set_driver_error(result, code, "zeCommandQueueExecuteCommandLists");
    goto done;
  }
  code = ze->fence_synchronize(fence, INFER_FENCE_TIMEOUT_NS);
  if (code != NPUNLOCK_ZE_SUCCESS) {
    set_driver_error(result, code, "zeFenceHostSynchronize");
    goto done;
  }
  success = true;

done:
  if (fence != NULL) {
    ze->fence_destroy(fence);
  }
  if (list != NULL) {
    ze->command_list_destroy(list);
  }
  if (queue != NULL) {
    ze->command_queue_destroy(queue);
  }
  return success;
}

static bool bind_arguments(const graph_functions *graph, npunlock_ze_graph_handle handle,
                           graph_argument *arguments, size_t argument_count, infer_result *result,
                           const char *stage) {
  size_t index;
  for (index = 0; index < argument_count; ++index) {
    uint32_t code =
        graph->set_argument(handle, arguments[index].index, arguments[index].allocation);
    if (code != NPUNLOCK_ZE_SUCCESS) {
      set_driver_error(result, code, stage);
      return false;
    }
  }
  return true;
}

void npunlock_infer_execute(const npunlock_infer_request *request, npunlock_infer_result *result) {
  infer_library loader = NULL;
  ze_functions ze = {0};
  graph_functions graph = {0};
  npunlock_ze_driver_handle driver = NULL;
  npunlock_ze_device_handle device = NULL;
  npunlock_ze_context_handle context = NULL;
  npunlock_ze_graph_handle handle = NULL;
  npunlock_ze_driver_properties driver_properties = {0};
  npunlock_ze_context_desc context_desc = {0};
  npunlock_ze_graph_desc_2 graph_desc = {0};
  npunlock_ze_graph_properties_3 graph_properties = {0};
  graph_argument *arguments = NULL;
  size_t argument_count = 0;
  uint32_t graph_version;
  uint32_t queue_ordinal = 0;
  uint32_t code;
  size_t index;

  memset(result, 0, sizeof(*result));
  result->status = NPUNLOCK_INFER_DRIVER_FAILED;
  result->stage = NPUNLOCK_INFER_STAGE_LOADER;
  result->driver_index = INFER_AUTO_INDEX;
  result->device_index = INFER_AUTO_INDEX;
  loader = open_loader();
  if (loader == NULL) {
    result->status = NPUNLOCK_INFER_LOADER_NOT_FOUND;
    set_diagnostic(result, "Level Zero loader libze_loader.so.1 was not found");
    goto done;
  }
  if (!load_ze_functions(loader, &ze)) {
    result->status = NPUNLOCK_INFER_SYMBOL_MISSING;
    set_diagnostic(result, "Level Zero loader is missing a required execution export");
    goto done;
  }
  result->stage = NPUNLOCK_INFER_STAGE_DEVICE;
  if (!select_npu(request, &ze, result, &driver, &device)) {
    goto done;
  }
  driver_properties.stype = NPUNLOCK_ZE_STRUCTURE_TYPE_DRIVER_PROPERTIES;
  code = ze.driver_get_properties(driver, &driver_properties);
  if (code != NPUNLOCK_ZE_SUCCESS) {
    set_driver_error(result, code, "zeDriverGetProperties");
    goto done;
  }
  result->driver_version = driver_properties.driverVersion;
  result->stage = NPUNLOCK_INFER_STAGE_GRAPH_EXTENSION;
  graph_version = find_graph_version(&ze, driver, result);
  if (graph_version == 0 || !get_graph_functions(&ze, driver, graph_version, result, &graph)) {
    goto done;
  }
  result->stage = NPUNLOCK_INFER_STAGE_GRAPH_CREATE;
  context_desc.stype = NPUNLOCK_ZE_STRUCTURE_TYPE_CONTEXT_DESC;
  code = ze.context_create(driver, &context_desc, &context);
  if (code != NPUNLOCK_ZE_SUCCESS) {
    set_driver_error(result, code, "zeContextCreate");
    goto done;
  }
  graph_desc.stype = NPUNLOCK_ZE_STRUCTURE_TYPE_GRAPH_DESC_2;
  graph_desc.format = NPUNLOCK_ZE_GRAPH_FORMAT_NATIVE;
  graph_desc.inputSize = request->graph_size;
  graph_desc.pInput = request->graph;
  graph_desc.pBuildFlags = NULL;
  code = graph.create2(context, device, &graph_desc, &handle);
  if (code != NPUNLOCK_ZE_SUCCESS || handle == NULL) {
    set_driver_error(result, code, "pfnCreate2(native)");
    goto done;
  }
  result->stage = NPUNLOCK_INFER_STAGE_INITIALIZATION;
  if (!query_arguments(request, &graph, handle, result, &arguments, &argument_count) ||
      !find_compute_queue(&ze, device, result, &queue_ordinal)) {
    goto done;
  }
  for (index = 0; index < argument_count; ++index) {
    graph_argument *argument = &arguments[index];
    npunlock_ze_host_mem_alloc_desc allocation_desc = {0};
    size_t allocation_size;
    allocation_desc.stype = NPUNLOCK_ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC;
    if (argument->type == NPUNLOCK_ZE_GRAPH_ARGUMENT_TYPE_INPUT) {
      allocation_desc.flags = NPUNLOCK_ZE_HOST_MEM_ALLOC_WRITE_COMBINED;
    }
    if (argument->data_size > SIZE_MAX - (INFER_PAGE_SIZE - 1)) {
      result->status = NPUNLOCK_INFER_UNSUPPORTED;
      set_diagnostic(result, "graph tensor allocation size overflowed");
      goto done;
    }
    allocation_size = (argument->data_size + INFER_PAGE_SIZE - 1) & ~(size_t)(INFER_PAGE_SIZE - 1);
    code = ze.mem_alloc_host(context, &allocation_desc, allocation_size, INFER_PAGE_SIZE,
                             &argument->allocation);
    if (code != NPUNLOCK_ZE_SUCCESS) {
      set_driver_error(result, code, "zeMemAllocHost(graph argument)");
      goto done;
    }
    memset(argument->allocation, 0, allocation_size);
    if (argument->type == NPUNLOCK_ZE_GRAPH_ARGUMENT_TYPE_INPUT) {
      const infer_request_input *input = &request->inputs[argument->request_input_index];
      memcpy(argument->allocation, input->data, input->data_size);
    } else if (argument->precision == NPUNLOCK_ZE_GRAPH_ARGUMENT_PRECISION_FP16) {
      size_t element;
      for (element = 0; element < argument->data_size / sizeof(uint16_t); ++element) {
        ((uint16_t *)argument->allocation)[element] = 0x7e00u;
      }
    } else {
      size_t element;
      for (element = 0; element < argument->data_size / sizeof(uint32_t); ++element) {
        ((uint32_t *)argument->allocation)[element] = 0x7fc00000u;
      }
    }
  }
  if (!bind_arguments(&graph, handle, arguments, argument_count, result,
                      "pfnSetArgumentValue(pre-init)")) {
    goto done;
  }
  graph_properties.stype = NPUNLOCK_ZE_STRUCTURE_TYPE_GRAPH_PROPERTIES_3;
  code = graph.get_properties3(handle, &graph_properties);
  if (code != NPUNLOCK_ZE_SUCCESS) {
    set_driver_error(result, code, "pfnGetProperties3(execute)");
    goto done;
  }
  if ((graph_properties.initStageRequired & NPUNLOCK_ZE_GRAPH_STAGE_INITIALIZE) != 0) {
    if (graph.initialize == NULL) {
      result->status = NPUNLOCK_INFER_UNSUPPORTED;
      set_diagnostic(result, "graph requires unavailable direct initialization");
      goto done;
    }
    code = graph.initialize(handle);
    if (code != NPUNLOCK_ZE_SUCCESS) {
      set_driver_error(result, code, "pfnGraphInitialize");
      goto done;
    }
  }
  if ((graph_properties.initStageRequired & NPUNLOCK_ZE_GRAPH_STAGE_COMMAND_LIST_INITIALIZE) != 0) {
    if (graph.append_initialize == NULL) {
      result->status = NPUNLOCK_INFER_UNSUPPORTED;
      set_diagnostic(result, "graph requires unavailable command-list initialization");
      goto done;
    }
    if (!submit(&ze, &graph, context, device, handle, queue_ordinal, true, result)) {
      goto done;
    }
  }
  if (!bind_arguments(&graph, handle, arguments, argument_count, result,
                      "pfnSetArgumentValue(post-init)")) {
    goto done;
  }
  result->stage = NPUNLOCK_INFER_STAGE_EXECUTION;
  if (!submit(&ze, &graph, context, device, handle, queue_ordinal, false, result)) {
    goto done;
  }
  for (index = 0; index < argument_count; ++index) {
    graph_argument *argument = &arguments[index];
    infer_result_output *output;
    if (argument->type != NPUNLOCK_ZE_GRAPH_ARGUMENT_TYPE_OUTPUT) {
      continue;
    }
    output = &result->outputs[result->output_count++];
    output->argument_index = argument->index;
    output->precision = argument->precision;
    output->dims_count = argument->dims_count;
    memcpy(output->dims, argument->dims, sizeof(output->dims));
    output->name_size = argument->name_size;
    memcpy(output->name, argument->name, output->name_size);
    output->data = (uint8_t *)malloc(argument->data_size);
    if (output->data == NULL) {
      result->status = NPUNLOCK_INFER_OUT_OF_MEMORY;
      goto done;
    }
    output->data_size = argument->data_size;
    memcpy(output->data, argument->allocation, argument->data_size);
  }
  result->status = NPUNLOCK_INFER_OK;
  result->stage = NPUNLOCK_INFER_STAGE_COMPLETE;
  result->driver_result = NPUNLOCK_ZE_SUCCESS;

done:
  if (arguments != NULL && context != NULL) {
    for (index = 0; index < argument_count; ++index) {
      if (arguments[index].allocation != NULL) {
        ze.mem_free(context, arguments[index].allocation);
      }
    }
  }
  free(arguments);
  if (handle != NULL && graph.destroy != NULL) {
    graph.destroy(handle);
  }
  if (context != NULL) {
    ze.context_destroy(context);
  }
  if (result->status != NPUNLOCK_INFER_OK) {
    for (index = 0; index < result->output_count; ++index) {
      free(result->outputs[index].data);
      result->outputs[index].data = NULL;
    }
    result->output_count = 0;
  }
  if (loader != NULL) {
    close_loader(loader);
  }
}

void npunlock_infer_result_release(npunlock_infer_result *result) {
  size_t index;
  if (result == NULL) {
    return;
  }
  for (index = 0; index < result->output_count; ++index) {
    free(result->outputs[index].data);
  }
  free(result->diagnostic);
  memset(result, 0, sizeof(*result));
}

const char *npunlock_infer_stage_name(npunlock_infer_stage stage) {
  switch (stage) {
  case NPUNLOCK_INFER_STAGE_LOADER:
    return "loader";
  case NPUNLOCK_INFER_STAGE_DEVICE:
    return "device-selection";
  case NPUNLOCK_INFER_STAGE_GRAPH_EXTENSION:
    return "graph-extension";
  case NPUNLOCK_INFER_STAGE_GRAPH_CREATE:
    return "graph-create";
  case NPUNLOCK_INFER_STAGE_INITIALIZATION:
    return "initialization";
  case NPUNLOCK_INFER_STAGE_EXECUTION:
    return "execution";
  case NPUNLOCK_INFER_STAGE_COMPLETE:
    return "complete";
  default:
    return "unknown";
  }
}
