#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "level_zero_min.h"
#include "run_protocol.h"

#define RUN_AUTO_INDEX UINT32_MAX
#define RUN_PAGE_SIZE 4096u
#define RUN_FENCE_TIMEOUT_NS 5000000000ull

typedef struct run_request {
  uint32_t driver_index;
  uint32_t device_index;
  const uint8_t *graph;
  size_t graph_size;
} run_request;

typedef struct run_result {
  npunlock_run_worker_status status;
  uint32_t driver_result;
  uint32_t driver_index;
  uint32_t device_index;
  uint32_t driver_version;
  uint32_t vendor_id;
  uint32_t device_id;
  uint64_t element_count;
  uint8_t *output;
  size_t output_size;
  char *diagnostic;
} run_result;

typedef struct graph_argument {
  uint32_t index;
  uint32_t dims_count;
  uint32_t dims[5];
} graph_argument;

typedef struct ze_functions {
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

static uint32_t load_u32(const uint8_t *data) {
  return (uint32_t)data[0] | ((uint32_t)data[1] << 8) | ((uint32_t)data[2] << 16) |
         ((uint32_t)data[3] << 24);
}

static uint64_t load_u64(const uint8_t *data) {
  return (uint64_t)load_u32(data) | ((uint64_t)load_u32(data + 4) << 32);
}

static void store_u32(uint8_t *data, uint32_t value) {
  data[0] = (uint8_t)value;
  data[1] = (uint8_t)(value >> 8);
  data[2] = (uint8_t)(value >> 16);
  data[3] = (uint8_t)(value >> 24);
}

static void store_u64(uint8_t *data, uint64_t value) {
  store_u32(data, (uint32_t)value);
  store_u32(data + 4, (uint32_t)(value >> 32));
}

static bool write_all(HANDLE handle, const uint8_t *data, size_t size) {
  while (size != 0) {
    DWORD chunk = size > MAXDWORD ? MAXDWORD : (DWORD)size;
    DWORD written = 0;
    if (!WriteFile(handle, data, chunk, &written, NULL) || written == 0) {
      return false;
    }
    data += written;
    size -= written;
  }
  return true;
}

static bool read_request(HANDLE handle, uint8_t **data, size_t *size) {
  uint8_t *buffer = NULL;
  size_t used = 0;
  size_t capacity = 0;
  const size_t maximum = NPUNLOCK_RUN_REQUEST_HEADER_SIZE + NPUNLOCK_RUN_MAX_GRAPH_SIZE;
  for (;;) {
    uint8_t chunk[16384];
    DWORD received = 0;
    if (!ReadFile(handle, chunk, sizeof(chunk), &received, NULL)) {
      if (GetLastError() == ERROR_BROKEN_PIPE) {
        break;
      }
      free(buffer);
      return false;
    }
    if (received == 0) {
      break;
    }
    if (used > maximum - received) {
      free(buffer);
      return false;
    }
    if (used + received > capacity) {
      size_t next = capacity == 0 ? 16384 : capacity;
      uint8_t *replacement;
      while (next < used + received) {
        if (next > maximum / 2) {
          next = maximum;
          break;
        }
        next *= 2;
      }
      replacement = (uint8_t *)realloc(buffer, next);
      if (replacement == NULL) {
        free(buffer);
        return false;
      }
      buffer = replacement;
      capacity = next;
    }
    memcpy(buffer + used, chunk, received);
    used += received;
  }
  *data = buffer;
  *size = used;
  return true;
}

static bool parse_request(const uint8_t *data, size_t size, run_request *request) {
  uint64_t graph_size;
  memset(request, 0, sizeof(*request));
  if (size < NPUNLOCK_RUN_REQUEST_HEADER_SIZE || load_u32(data) != NPUNLOCK_RUN_REQUEST_MAGIC ||
      load_u32(data + 4) != NPUNLOCK_RUN_PROTOCOL_VERSION || load_u64(data + 24) != 0) {
    return false;
  }
  graph_size = load_u64(data + 16);
  if (graph_size == 0 || graph_size > NPUNLOCK_RUN_MAX_GRAPH_SIZE ||
      graph_size != size - NPUNLOCK_RUN_REQUEST_HEADER_SIZE) {
    return false;
  }
  request->driver_index = load_u32(data + 8);
  request->device_index = load_u32(data + 12);
  request->graph = data + NPUNLOCK_RUN_REQUEST_HEADER_SIZE;
  request->graph_size = (size_t)graph_size;
  return true;
}

static void set_diagnostic(run_result *result, const char *message) {
  size_t size;
  if (result->diagnostic != NULL || message == NULL) {
    return;
  }
  size = strlen(message);
  if (size > NPUNLOCK_RUN_MAX_DIAGNOSTIC_SIZE) {
    size = NPUNLOCK_RUN_MAX_DIAGNOSTIC_SIZE;
  }
  result->diagnostic = (char *)malloc(size + 1);
  if (result->diagnostic != NULL) {
    memcpy(result->diagnostic, message, size);
    result->diagnostic[size] = '\0';
  }
}

static void set_driver_error(run_result *result, uint32_t code, const char *stage) {
  char message[192];
  result->driver_result = code;
  result->status = NPUNLOCK_RUN_WORKER_DRIVER_FAILED;
  snprintf(message, sizeof(message), "%s returned Level Zero result 0x%08x", stage, code);
  set_diagnostic(result, message);
}

static bool load_export(HMODULE library, const char *name, void *destination, size_t size) {
  FARPROC proc = GetProcAddress(library, name);
  if (proc == NULL || size != sizeof(proc)) {
    return false;
  }
  memcpy(destination, &proc, size);
  return true;
}

static bool load_ze_functions(HMODULE library, ze_functions *ze) {
  memset(ze, 0, sizeof(*ze));
#define LOAD(name, field) load_export(library, name, &ze->field, sizeof(ze->field))
  return LOAD("zeInitDrivers", init_drivers) &&
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
         LOAD("zeCommandListDestroy", command_list_destroy) &&
         LOAD("zeFenceCreate", fence_create) && LOAD("zeFenceDestroy", fence_destroy) &&
         LOAD("zeFenceHostSynchronize", fence_synchronize);
#undef LOAD
}

static bool select_npu(const run_request *request, const ze_functions *ze, run_result *result,
                       npunlock_ze_driver_handle *selected_driver,
                       npunlock_ze_device_handle *selected_device) {
  npunlock_ze_init_driver_type_desc init = {NPUNLOCK_ZE_STRUCTURE_TYPE_INIT_DRIVER_TYPE_DESC, NULL,
                                            NPUNLOCK_ZE_INIT_DRIVER_TYPE_FLAG_NPU};
  npunlock_ze_driver_handle *drivers = NULL;
  uint32_t driver_count = 0;
  uint32_t driver_index;
  uint32_t code = ze->init_drivers(&driver_count, NULL, &init);
  if (code != NPUNLOCK_ZE_SUCCESS) {
    set_driver_error(result, code, "zeInitDrivers(count)");
    return false;
  }
  if (driver_count == 0 ||
      (request->driver_index != RUN_AUTO_INDEX && request->driver_index >= driver_count)) {
    result->status = NPUNLOCK_RUN_WORKER_NPU_NOT_FOUND;
    set_diagnostic(result, "requested NPU Level Zero driver was not found");
    return false;
  }
  drivers = (npunlock_ze_driver_handle *)calloc(driver_count, sizeof(*drivers));
  if (drivers == NULL) {
    result->status = NPUNLOCK_RUN_WORKER_OUT_OF_MEMORY;
    return false;
  }
  code = ze->init_drivers(&driver_count, drivers, &init);
  if (code != NPUNLOCK_ZE_SUCCESS) {
    set_driver_error(result, code, "zeInitDrivers(list)");
    free(drivers);
    return false;
  }
  for (driver_index = 0; driver_index < driver_count; ++driver_index) {
    npunlock_ze_device_handle *devices = NULL;
    uint32_t device_count = 0;
    uint32_t device_index;
    if (request->driver_index != RUN_AUTO_INDEX && request->driver_index != driver_index) {
      continue;
    }
    code = ze->device_get(drivers[driver_index], &device_count, NULL);
    if (code != NPUNLOCK_ZE_SUCCESS || device_count == 0) {
      continue;
    }
    devices = (npunlock_ze_device_handle *)calloc(device_count, sizeof(*devices));
    if (devices == NULL) {
      result->status = NPUNLOCK_RUN_WORKER_OUT_OF_MEMORY;
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
      if (request->device_index != RUN_AUTO_INDEX && request->device_index != device_index) {
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
  result->status = NPUNLOCK_RUN_WORKER_NPU_NOT_FOUND;
  set_diagnostic(result, "requested Level Zero VPU device was not found");
  return false;
}

static uint32_t find_graph_version(const ze_functions *ze, npunlock_ze_driver_handle driver,
                                   run_result *result) {
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
    result->status = NPUNLOCK_RUN_WORKER_OUT_OF_MEMORY;
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
    result->status = NPUNLOCK_RUN_WORKER_GRAPH_EXTENSION_MISSING;
    set_diagnostic(result, "selected NPU driver does not advertise ZE_extension_graph");
  }
  return version;
}

static bool get_graph_functions(const ze_functions *ze, npunlock_ze_driver_handle driver,
                                uint32_t version, run_result *result, graph_functions *graph) {
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
    result->status = NPUNLOCK_RUN_WORKER_UNSUPPORTED;
    set_diagnostic(result, "required native graph execution entry point is unavailable");
    return false;
  }
  return true;
}

static bool argument_element_count(const npunlock_ze_graph_argument_properties_3 *properties,
                                   uint64_t *count) {
  uint64_t product = 1;
  uint32_t index;
  if (properties->dims_count == 0 || properties->dims_count > 5) {
    return false;
  }
  for (index = 0; index < properties->dims_count; ++index) {
    if (properties->dims[index] == 0 || product > UINT64_MAX / properties->dims[index]) {
      return false;
    }
    product *= properties->dims[index];
  }
  *count = product;
  return true;
}

static bool query_arguments(const graph_functions *graph, npunlock_ze_graph_handle handle,
                            run_result *result, graph_argument *input, graph_argument *output,
                            uint64_t *element_count) {
  npunlock_ze_graph_properties_3 properties = {0};
  bool have_input = false;
  bool have_output = false;
  uint64_t input_count = 0;
  uint64_t output_count = 0;
  uint32_t index;
  uint32_t code;
  properties.stype = NPUNLOCK_ZE_STRUCTURE_TYPE_GRAPH_PROPERTIES_3;
  code = graph->get_properties3(handle, &properties);
  if (code != NPUNLOCK_ZE_SUCCESS) {
    set_driver_error(result, code, "pfnGetProperties3");
    return false;
  }
  if (properties.numGraphArgs != 2) {
    result->status = NPUNLOCK_RUN_WORKER_UNSUPPORTED;
    set_diagnostic(result, "execution requires exactly one input and one output argument");
    return false;
  }
  for (index = 0; index < properties.numGraphArgs; ++index) {
    npunlock_ze_graph_argument_properties_3 argument = {0};
    graph_argument selected = {0};
    uint64_t count;
    argument.stype = NPUNLOCK_ZE_STRUCTURE_TYPE_GRAPH_ARGUMENT_PROPERTIES_3;
    code = graph->get_argument_properties3(handle, index, &argument);
    if (code != NPUNLOCK_ZE_SUCCESS) {
      set_driver_error(result, code, "pfnGetArgumentProperties3");
      return false;
    }
    if (argument.devicePrecision != NPUNLOCK_ZE_GRAPH_ARGUMENT_PRECISION_FP16 ||
        !argument_element_count(&argument, &count)) {
      result->status = NPUNLOCK_RUN_WORKER_UNSUPPORTED;
      set_diagnostic(result, "execution requires bounded static FP16 graph arguments");
      return false;
    }
    selected.index = index;
    selected.dims_count = argument.dims_count;
    memcpy(selected.dims, argument.dims, sizeof(selected.dims));
    if (argument.type == NPUNLOCK_ZE_GRAPH_ARGUMENT_TYPE_INPUT && !have_input) {
      *input = selected;
      input_count = count;
      have_input = true;
    } else if (argument.type == NPUNLOCK_ZE_GRAPH_ARGUMENT_TYPE_OUTPUT && !have_output) {
      *output = selected;
      output_count = count;
      have_output = true;
    } else {
      result->status = NPUNLOCK_RUN_WORKER_UNSUPPORTED;
      set_diagnostic(result, "execution graph argument roles are unsupported");
      return false;
    }
  }
  if (!have_input || !have_output || input_count != output_count ||
      input->dims_count != output->dims_count ||
      memcmp(input->dims, output->dims, input->dims_count * sizeof(input->dims[0])) != 0 ||
      input_count > NPUNLOCK_RUN_MAX_OUTPUT_SIZE / sizeof(uint16_t)) {
    result->status = NPUNLOCK_RUN_WORKER_UNSUPPORTED;
    set_diagnostic(result, "execution requires matching bounded input and output shapes");
    return false;
  }
  *element_count = input_count;
  return true;
}

static uint16_t float_to_fp16(float value) {
  uint32_t bits;
  uint16_t sign;
  uint32_t exponent_bits;
  uint32_t mantissa;
  int exponent;
  memcpy(&bits, &value, sizeof(bits));
  sign = (uint16_t)((bits >> 16) & 0x8000u);
  exponent_bits = (bits >> 23) & 0xffu;
  mantissa = bits & 0x7fffffu;
  if (exponent_bits == 0xffu) {
    uint16_t payload = (uint16_t)(mantissa >> 13);
    return mantissa == 0 ? (uint16_t)(sign | 0x7c00u)
                         : (uint16_t)(sign | 0x7c00u | (payload == 0 ? 1 : payload));
  }
  exponent = (int)exponent_bits - 127 + 15;
  if (exponent >= 31) {
    return (uint16_t)(sign | 0x7c00u);
  }
  if (exponent <= 0) {
    uint32_t rounded;
    uint32_t remainder;
    uint32_t halfway;
    unsigned shift;
    if (exponent < -10) {
      return sign;
    }
    mantissa |= 0x800000u;
    shift = (unsigned)(14 - exponent);
    rounded = mantissa >> shift;
    remainder = mantissa & ((1u << shift) - 1u);
    halfway = 1u << (shift - 1u);
    if (remainder > halfway || (remainder == halfway && (rounded & 1u) != 0)) {
      ++rounded;
    }
    return (uint16_t)(sign | rounded);
  }
  {
    uint32_t rounded = mantissa >> 13;
    uint32_t remainder = mantissa & 0x1fffu;
    if (remainder > 0x1000u || (remainder == 0x1000u && (rounded & 1u) != 0)) {
      ++rounded;
      if (rounded == 0x400u) {
        rounded = 0;
        ++exponent;
        if (exponent >= 31) {
          return (uint16_t)(sign | 0x7c00u);
        }
      }
    }
    return (uint16_t)(sign | ((uint16_t)exponent << 10) | rounded);
  }
}

static bool find_compute_queue(const ze_functions *ze, npunlock_ze_device_handle device,
                               run_result *result, uint32_t *ordinal) {
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
    result->status = NPUNLOCK_RUN_WORKER_OUT_OF_MEMORY;
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
  result->status = NPUNLOCK_RUN_WORKER_UNSUPPORTED;
  set_diagnostic(result, "selected NPU exposes no compute command queue group");
  return false;
}

static bool submit(const ze_functions *ze, const graph_functions *graph,
                   npunlock_ze_context_handle context, npunlock_ze_device_handle device,
                   npunlock_ze_graph_handle handle, uint32_t ordinal, bool initialize,
                   run_result *result) {
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
  code = ze->fence_synchronize(fence, RUN_FENCE_TIMEOUT_NS);
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

static void execute_graph(const run_request *request, run_result *result) {
  HMODULE loader = NULL;
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
  npunlock_ze_host_mem_alloc_desc input_desc = {0};
  npunlock_ze_host_mem_alloc_desc output_desc = {0};
  graph_argument input_argument = {0};
  graph_argument output_argument = {0};
  void *input = NULL;
  void *output = NULL;
  uint64_t element_count = 0;
  size_t tensor_bytes;
  size_t allocation_bytes;
  uint32_t graph_version;
  uint32_t queue_ordinal = 0;
  uint32_t code;
  size_t index;

  memset(result, 0, sizeof(*result));
  result->status = NPUNLOCK_RUN_WORKER_DRIVER_FAILED;
  loader = LoadLibraryExW(L"ze_loader.dll", NULL, LOAD_LIBRARY_SEARCH_SYSTEM32);
  if (loader == NULL) {
    result->status = NPUNLOCK_RUN_WORKER_LOADER_NOT_FOUND;
    set_diagnostic(result, "Level Zero loader ze_loader.dll was not found in System32");
    goto done;
  }
  if (!load_ze_functions(loader, &ze)) {
    result->status = NPUNLOCK_RUN_WORKER_SYMBOL_MISSING;
    set_diagnostic(result, "Level Zero loader is missing a required execution export");
    goto done;
  }
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
  graph_version = find_graph_version(&ze, driver, result);
  if (graph_version == 0 || !get_graph_functions(&ze, driver, graph_version, result, &graph)) {
    goto done;
  }
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
  if (!query_arguments(&graph, handle, result, &input_argument, &output_argument, &element_count) ||
      !find_compute_queue(&ze, device, result, &queue_ordinal)) {
    goto done;
  }
  tensor_bytes = (size_t)element_count * sizeof(uint16_t);
  if (tensor_bytes == 0 || tensor_bytes > SIZE_MAX - (RUN_PAGE_SIZE - 1)) {
    result->status = NPUNLOCK_RUN_WORKER_UNSUPPORTED;
    set_diagnostic(result, "graph tensor size is outside the execution limit");
    goto done;
  }
  allocation_bytes = (tensor_bytes + RUN_PAGE_SIZE - 1) & ~(size_t)(RUN_PAGE_SIZE - 1);
  input_desc.stype = NPUNLOCK_ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC;
  input_desc.flags = NPUNLOCK_ZE_HOST_MEM_ALLOC_WRITE_COMBINED;
  output_desc.stype = NPUNLOCK_ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC;
  code = ze.mem_alloc_host(context, &input_desc, allocation_bytes, RUN_PAGE_SIZE, &input);
  if (code != NPUNLOCK_ZE_SUCCESS) {
    set_driver_error(result, code, "zeMemAllocHost(input)");
    goto done;
  }
  code = ze.mem_alloc_host(context, &output_desc, allocation_bytes, RUN_PAGE_SIZE, &output);
  if (code != NPUNLOCK_ZE_SUCCESS) {
    set_driver_error(result, code, "zeMemAllocHost(output)");
    goto done;
  }
  memset(input, 0, allocation_bytes);
  memset(output, 0, allocation_bytes);
  for (index = 0; index < (size_t)element_count; ++index) {
    float step = element_count > 1 ? 3.75f / (float)(element_count - 1) : 0.0f;
    ((uint16_t *)input)[index] = float_to_fp16(-2.0f + (float)index * step);
    ((uint16_t *)output)[index] = 0x7e00u;
  }
  code = graph.set_argument(handle, input_argument.index, input);
  if (code != NPUNLOCK_ZE_SUCCESS) {
    set_driver_error(result, code, "pfnSetArgumentValue(input)");
    goto done;
  }
  code = graph.set_argument(handle, output_argument.index, output);
  if (code != NPUNLOCK_ZE_SUCCESS) {
    set_driver_error(result, code, "pfnSetArgumentValue(output)");
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
      result->status = NPUNLOCK_RUN_WORKER_UNSUPPORTED;
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
      result->status = NPUNLOCK_RUN_WORKER_UNSUPPORTED;
      set_diagnostic(result, "graph requires unavailable command-list initialization");
      goto done;
    }
    if (!submit(&ze, &graph, context, device, handle, queue_ordinal, true, result)) {
      goto done;
    }
  }
  code = graph.set_argument(handle, input_argument.index, input);
  if (code != NPUNLOCK_ZE_SUCCESS) {
    set_driver_error(result, code, "pfnSetArgumentValue(input post-init)");
    goto done;
  }
  code = graph.set_argument(handle, output_argument.index, output);
  if (code != NPUNLOCK_ZE_SUCCESS) {
    set_driver_error(result, code, "pfnSetArgumentValue(output post-init)");
    goto done;
  }
  if (!submit(&ze, &graph, context, device, handle, queue_ordinal, false, result)) {
    goto done;
  }
  result->output = (uint8_t *)malloc(tensor_bytes);
  if (result->output == NULL) {
    result->status = NPUNLOCK_RUN_WORKER_OUT_OF_MEMORY;
    goto done;
  }
  memcpy(result->output, output, tensor_bytes);
  result->output_size = tensor_bytes;
  result->element_count = element_count;
  result->status = NPUNLOCK_RUN_WORKER_OK;
  result->driver_result = NPUNLOCK_ZE_SUCCESS;

done:
  if (output != NULL) {
    ze.mem_free(context, output);
  }
  if (input != NULL) {
    ze.mem_free(context, input);
  }
  if (handle != NULL && graph.destroy != NULL) {
    graph.destroy(handle);
  }
  if (context != NULL) {
    ze.context_destroy(context);
  }
  if (result->status != NPUNLOCK_RUN_WORKER_OK) {
    free(result->output);
    result->output = NULL;
    result->output_size = 0;
    result->element_count = 0;
  }
  if (loader != NULL) {
    FreeLibrary(loader);
  }
}

static bool send_response(HANDLE handle, const run_result *result) {
  uint8_t header[NPUNLOCK_RUN_RESPONSE_HEADER_SIZE] = {0};
  size_t diagnostic_size = result->diagnostic == NULL ? 0 : strlen(result->diagnostic);
  if (result->output_size > UINT32_MAX || diagnostic_size > UINT32_MAX) {
    return false;
  }
  store_u32(header, NPUNLOCK_RUN_RESPONSE_MAGIC);
  store_u32(header + 4, NPUNLOCK_RUN_PROTOCOL_VERSION);
  store_u32(header + 8, (uint32_t)result->status);
  store_u32(header + 12, result->driver_result);
  store_u32(header + 16, result->driver_index);
  store_u32(header + 20, result->device_index);
  store_u32(header + 24, result->driver_version);
  store_u32(header + 28, result->vendor_id);
  store_u32(header + 32, result->device_id);
  store_u64(header + 40, result->element_count);
  store_u32(header + 48, (uint32_t)result->output_size);
  store_u32(header + 52, (uint32_t)diagnostic_size);
  return write_all(handle, header, sizeof(header)) &&
         write_all(handle, result->output, result->output_size) &&
         write_all(handle, (const uint8_t *)result->diagnostic, diagnostic_size);
}

int main(int argc, char **argv) {
  char *handle_end = NULL;
  uint64_t handle_value;
  HANDLE response;
  HANDLE input = GetStdHandle(STD_INPUT_HANDLE);
  uint8_t *request_data = NULL;
  size_t request_size = 0;
  run_request request;
  run_result result;
  int exit_code;

  SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX | SEM_NOOPENFILEERRORBOX);
  if (argc != 3 || strcmp(argv[1], "--response-handle") != 0) {
    return 2;
  }
  handle_value = _strtoui64(argv[2], &handle_end, 10);
  if (handle_value == 0 || handle_end == argv[2] || *handle_end != '\0') {
    return 2;
  }
  response = (HANDLE)(uintptr_t)handle_value;
  memset(&result, 0, sizeof(result));
  result.status = NPUNLOCK_RUN_WORKER_BAD_REQUEST;
  if (input == NULL || input == INVALID_HANDLE_VALUE ||
      !read_request(input, &request_data, &request_size)) {
    set_diagnostic(&result, "failed to read execution worker request");
  } else if (!parse_request(request_data, request_size, &request)) {
    set_diagnostic(&result, "malformed execution worker request");
  } else {
    execute_graph(&request, &result);
  }
  exit_code = send_response(response, &result) ? 0 : 3;
  free(result.output);
  free(result.diagnostic);
  free(request_data);
  CloseHandle(response);
  return exit_code;
}
