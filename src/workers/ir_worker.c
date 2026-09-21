#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ir_protocol.h"
#include "level_zero_min.h"
#include "worker_entry.h"
#include "worker_io.h"
#include "worker_protocol.h"

#define checked_add npunlock_worker_checked_add
#define load_u32 npunlock_worker_load_u32
#define load_u64 npunlock_worker_load_u64
#define store_u32 npunlock_worker_store_u32
#define store_u64 npunlock_worker_store_u64
#define write_all npunlock_worker_write_all

#define IR2BLOB_AUTO_INDEX UINT32_MAX

typedef struct ir_request {
  uint32_t driver_index;
  uint32_t device_index;
  const uint8_t *xml;
  size_t xml_size;
  const uint8_t *weights;
  size_t weights_size;
  const uint8_t *build_flags;
  size_t build_flags_size;
} ir_request;

typedef struct ir_result {
  npunlock_ir_worker_status status;
  uint32_t driver_result;
  uint32_t driver_index;
  uint32_t device_index;
  uint32_t graph_version;
  uint16_t compiler_major;
  uint16_t compiler_minor;
  uint32_t max_opset;
  uint32_t driver_version;
  uint32_t vendor_id;
  uint32_t device_id;
  npunlock_ze_graph_version elf_version;
  npunlock_ze_graph_version runtime_version;
  uint8_t *blob;
  size_t blob_size;
  char *diagnostic;
} ir_result;

typedef struct ze_functions {
  npunlock_ze_init_drivers_fn init_drivers;
  npunlock_ze_driver_get_properties_fn driver_get_properties;
  npunlock_ze_driver_get_extensions_fn driver_get_extensions;
  npunlock_ze_driver_get_extension_address_fn driver_get_extension_address;
  npunlock_ze_device_get_fn device_get;
  npunlock_ze_device_get_properties_fn device_get_properties;
  npunlock_ze_context_create_fn context_create;
  npunlock_ze_context_destroy_fn context_destroy;
} ze_functions;

_Static_assert(offsetof(npunlock_ze_device_properties, type) == 16,
               "Level Zero device property ABI mismatch");
_Static_assert(offsetof(npunlock_ze_device_properties, name) == 112,
               "Level Zero device property ABI mismatch");
_Static_assert(sizeof(npunlock_ze_device_properties) == 368,
               "Level Zero device property ABI mismatch");
_Static_assert(sizeof(npunlock_ze_graph_desc_2) == 56, "Level Zero graph descriptor ABI mismatch");

static bool parse_request(const uint8_t *data, size_t size, ir_request *request) {
  uint64_t xml_size;
  uint64_t weights_size;
  uint32_t flags_size;
  size_t expected = NPUNLOCK_IR_REQUEST_HEADER_SIZE;
  size_t offset;
  memset(request, 0, sizeof(*request));
  if (size < NPUNLOCK_IR_REQUEST_HEADER_SIZE || load_u32(data) != NPUNLOCK_IR_REQUEST_MAGIC ||
      load_u32(data + 4) != NPUNLOCK_IR_PROTOCOL_VERSION || load_u32(data + 36) != 0 ||
      load_u64(data + 40) != 0) {
    return false;
  }
  xml_size = load_u64(data + 16);
  weights_size = load_u64(data + 24);
  flags_size = load_u32(data + 32);
  if (xml_size == 0 || xml_size > SIZE_MAX || weights_size > SIZE_MAX ||
      !checked_add(expected, (size_t)xml_size, &expected) ||
      !checked_add(expected, (size_t)weights_size, &expected) ||
      !checked_add(expected, flags_size, &expected) || expected != size) {
    return false;
  }
  offset = NPUNLOCK_IR_REQUEST_HEADER_SIZE;
  request->driver_index = load_u32(data + 8);
  request->device_index = load_u32(data + 12);
  request->xml = data + offset;
  request->xml_size = (size_t)xml_size;
  offset += (size_t)xml_size;
  request->weights = data + offset;
  request->weights_size = (size_t)weights_size;
  offset += (size_t)weights_size;
  request->build_flags = data + offset;
  request->build_flags_size = flags_size;
  return memchr(request->build_flags, 0, request->build_flags_size) == NULL;
}

static void set_diagnostic(ir_result *result, const char *message) {
  size_t size;
  if (result->diagnostic != NULL || message == NULL) {
    return;
  }
  size = strlen(message);
  if (size > NPUNLOCK_IR_MAX_DIAGNOSTIC_SIZE) {
    size = NPUNLOCK_IR_MAX_DIAGNOSTIC_SIZE;
  }
  result->diagnostic = (char *)malloc(size + 1);
  if (result->diagnostic != NULL) {
    memcpy(result->diagnostic, message, size);
    result->diagnostic[size] = '\0';
  }
}

static void set_driver_error(ir_result *result, uint32_t code, const char *stage) {
  char message[192];
  result->driver_result = code;
  result->status = NPUNLOCK_IR_WORKER_DRIVER_FAILED;
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

static bool load_ze_functions(HMODULE library, ze_functions *functions) {
  memset(functions, 0, sizeof(*functions));
  return load_export(library, "zeInitDrivers", &functions->init_drivers,
                     sizeof(functions->init_drivers)) &&
         load_export(library, "zeDriverGetProperties", &functions->driver_get_properties,
                     sizeof(functions->driver_get_properties)) &&
         load_export(library, "zeDriverGetExtensionProperties", &functions->driver_get_extensions,
                     sizeof(functions->driver_get_extensions)) &&
         load_export(library, "zeDriverGetExtensionFunctionAddress",
                     &functions->driver_get_extension_address,
                     sizeof(functions->driver_get_extension_address)) &&
         load_export(library, "zeDeviceGet", &functions->device_get,
                     sizeof(functions->device_get)) &&
         load_export(library, "zeDeviceGetProperties", &functions->device_get_properties,
                     sizeof(functions->device_get_properties)) &&
         load_export(library, "zeContextCreate", &functions->context_create,
                     sizeof(functions->context_create)) &&
         load_export(library, "zeContextDestroy", &functions->context_destroy,
                     sizeof(functions->context_destroy));
}

static bool select_npu(const ir_request *request, const ze_functions *ze, ir_result *result,
                       npunlock_ze_driver_handle *selected_driver,
                       npunlock_ze_device_handle *selected_device) {
  npunlock_ze_init_driver_type_desc init = {NPUNLOCK_ZE_STRUCTURE_TYPE_INIT_DRIVER_TYPE_DESC, NULL,
                                            NPUNLOCK_ZE_INIT_DRIVER_TYPE_FLAG_NPU};
  npunlock_ze_driver_handle *drivers = NULL;
  uint32_t driver_count = 0;
  uint32_t code;
  uint32_t driver_index;
  code = ze->init_drivers(&driver_count, NULL, &init);
  if (code != NPUNLOCK_ZE_SUCCESS) {
    set_driver_error(result, code, "zeInitDrivers(count)");
    return false;
  }
  if (driver_count == 0 ||
      (request->driver_index != IR2BLOB_AUTO_INDEX && request->driver_index >= driver_count)) {
    result->status = NPUNLOCK_IR_WORKER_NPU_NOT_FOUND;
    set_diagnostic(result, "requested NPU Level Zero driver was not found");
    return false;
  }
  drivers = (npunlock_ze_driver_handle *)calloc(driver_count, sizeof(*drivers));
  if (drivers == NULL) {
    result->status = NPUNLOCK_IR_WORKER_OUT_OF_MEMORY;
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
    if (request->driver_index != IR2BLOB_AUTO_INDEX && request->driver_index != driver_index) {
      continue;
    }
    code = ze->device_get(drivers[driver_index], &device_count, NULL);
    if (code != NPUNLOCK_ZE_SUCCESS || device_count == 0 ||
        (request->device_index != IR2BLOB_AUTO_INDEX && request->device_index >= device_count)) {
      continue;
    }
    devices = (npunlock_ze_device_handle *)calloc(device_count, sizeof(*devices));
    if (devices == NULL) {
      result->status = NPUNLOCK_IR_WORKER_OUT_OF_MEMORY;
      free(drivers);
      return false;
    }
    code = ze->device_get(drivers[driver_index], &device_count, devices);
    if (code != NPUNLOCK_ZE_SUCCESS) {
      free(devices);
      continue;
    }
    for (device_index = 0; device_index < device_count; ++device_index) {
      npunlock_ze_device_properties properties;
      if (request->device_index != IR2BLOB_AUTO_INDEX && request->device_index != device_index) {
        continue;
      }
      memset(&properties, 0, sizeof(properties));
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
  result->status = NPUNLOCK_IR_WORKER_NPU_NOT_FOUND;
  set_diagnostic(result, "requested Level Zero VPU device was not found");
  return false;
}

static uint32_t find_graph_version(const ze_functions *ze, npunlock_ze_driver_handle driver,
                                   ir_result *result) {
  npunlock_ze_driver_extension_properties *extensions = NULL;
  uint32_t count = 0;
  uint32_t code;
  uint32_t index;
  uint32_t version = 0;
  code = ze->driver_get_extensions(driver, &count, NULL);
  if (code != NPUNLOCK_ZE_SUCCESS) {
    set_driver_error(result, code, "zeDriverGetExtensionProperties(count)");
    return 0;
  }
  extensions = (npunlock_ze_driver_extension_properties *)calloc(count, sizeof(*extensions));
  if (extensions == NULL && count != 0) {
    result->status = NPUNLOCK_IR_WORKER_OUT_OF_MEMORY;
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
    result->status = NPUNLOCK_IR_WORKER_GRAPH_EXTENSION_MISSING;
    set_diagnostic(result, "selected NPU driver does not advertise ZE_extension_graph");
  }
  return version;
}

static bool get_graph_ddi(const ze_functions *ze, npunlock_ze_driver_handle driver,
                          uint32_t graph_version, ir_result *result,
                          npunlock_ze_graph_ddi_prefix **ddi) {
  void *raw_driver_ddi = NULL;
  npunlock_ze_npu_get_extension_fn get_extension = NULL;
  npunlock_ze_driver_extension_npu request;
  uint32_t code =
      ze->driver_get_extension_address(driver, "ZE_extension_driver_npu", &raw_driver_ddi);
  if (code != NPUNLOCK_ZE_SUCCESS || raw_driver_ddi == NULL) {
    set_driver_error(result, code, "zeDriverGetExtensionFunctionAddress");
    return false;
  }
  memcpy(&get_extension, raw_driver_ddi, sizeof(get_extension));
  if (get_extension == NULL) {
    result->status = NPUNLOCK_IR_WORKER_SYMBOL_MISSING;
    set_diagnostic(result, "NPU driver extension table is invalid");
    return false;
  }
  memset(&request, 0, sizeof(request));
  request.stype = NPUNLOCK_ZE_STRUCTURE_TYPE_DRIVER_EXTENSION_NPU_EXT;
  request.name = "ZE_extension_graph";
  request.version = graph_version;
  request.ppFunctionAddress = (void **)ddi;
  code = get_extension(driver, &request);
  if (code != NPUNLOCK_ZE_SUCCESS || *ddi == NULL) {
    set_driver_error(result, code, "ZE_extension_driver_npu::pfnGetExtension");
    return false;
  }
  return true;
}

static bool query_graph_contract(npunlock_ze_graph_ddi_prefix *ddi,
                                 npunlock_ze_device_handle device, ir_result *result) {
  npunlock_ze_graph_device_properties2_fn get2 = NULL;
  npunlock_ze_graph_device_properties_fn get1 = NULL;
  uint32_t code;
  memcpy(&get2, &ddi->slots[NPUNLOCK_ZE_GRAPH_DDI_DEVICE_PROPERTIES2], sizeof(get2));
  if (get2 != NULL) {
    npunlock_ze_device_graph_properties_2 properties;
    memset(&properties, 0, sizeof(properties));
    properties.stype = NPUNLOCK_ZE_STRUCTURE_TYPE_DEVICE_GRAPH_PROPERTIES_2;
    code = get2(device, &properties);
    if (code != NPUNLOCK_ZE_SUCCESS) {
      set_driver_error(result, code, "pfnDeviceGetGraphProperties2");
      return false;
    }
    result->compiler_major = properties.compilerVersion.major;
    result->compiler_minor = properties.compilerVersion.minor;
    result->max_opset = properties.maxOVOpsetVersionSupported;
    result->elf_version = properties.elfVersion;
    result->runtime_version = properties.runtimeVersion;
    if ((properties.graphFormatsSupported & NPUNLOCK_ZE_GRAPH_FORMAT_NGRAPH_LITE) == 0) {
      result->status = NPUNLOCK_IR_WORKER_UNSUPPORTED;
      set_diagnostic(result, "selected NPU does not advertise NGRAPH_LITE input");
      return false;
    }
    return true;
  }
  memcpy(&get1, &ddi->slots[NPUNLOCK_ZE_GRAPH_DDI_DEVICE_PROPERTIES], sizeof(get1));
  if (get1 != NULL) {
    npunlock_ze_device_graph_properties properties;
    memset(&properties, 0, sizeof(properties));
    properties.stype = NPUNLOCK_ZE_STRUCTURE_TYPE_DEVICE_GRAPH_PROPERTIES;
    code = get1(device, &properties);
    if (code != NPUNLOCK_ZE_SUCCESS) {
      set_driver_error(result, code, "pfnDeviceGetGraphProperties");
      return false;
    }
    result->compiler_major = properties.compilerVersion.major;
    result->compiler_minor = properties.compilerVersion.minor;
    result->max_opset = properties.maxOVOpsetVersionSupported;
    if ((properties.graphFormatsSupported & NPUNLOCK_ZE_GRAPH_FORMAT_NGRAPH_LITE) == 0) {
      result->status = NPUNLOCK_IR_WORKER_UNSUPPORTED;
      set_diagnostic(result, "selected NPU does not advertise NGRAPH_LITE input");
      return false;
    }
    return true;
  }
  result->status = NPUNLOCK_IR_WORKER_UNSUPPORTED;
  set_diagnostic(result, "graph device-property query is unavailable");
  return false;
}

static uint8_t *package_ir(const ir_request *request, const ir_result *result, size_t *size) {
  size_t total = 24;
  size_t offset = 0;
  uint8_t *data;
  if (!checked_add(total, request->xml_size, &total) ||
      !checked_add(total, request->weights_size, &total)) {
    return NULL;
  }
  data = (uint8_t *)malloc(total);
  if (data == NULL) {
    return NULL;
  }
  data[offset++] = (uint8_t)result->compiler_major;
  data[offset++] = (uint8_t)(result->compiler_major >> 8);
  data[offset++] = (uint8_t)result->compiler_minor;
  data[offset++] = (uint8_t)(result->compiler_minor >> 8);
  store_u32(data + offset, 2);
  offset += 4;
  store_u64(data + offset, request->xml_size);
  offset += 8;
  memcpy(data + offset, request->xml, request->xml_size);
  offset += request->xml_size;
  store_u64(data + offset, request->weights_size);
  offset += 8;
  if (request->weights_size != 0) {
    memcpy(data + offset, request->weights, request->weights_size);
  }
  *size = total;
  return data;
}

static void compile_ir(const ir_request *request, ir_result *result) {
  HMODULE loader = NULL;
  ze_functions ze = {0};
  npunlock_ze_driver_handle driver = NULL;
  npunlock_ze_device_handle device = NULL;
  npunlock_ze_graph_ddi_prefix *ddi = NULL;
  npunlock_ze_context_handle context = NULL;
  npunlock_ze_graph_handle graph = NULL;
  npunlock_ze_driver_properties driver_properties;
  npunlock_ze_context_desc context_desc;
  npunlock_ze_graph_desc_2 graph_desc;
  npunlock_ze_graph_create2_fn create2 = NULL;
  npunlock_ze_graph_destroy_fn destroy = NULL;
  npunlock_ze_graph_get_native2_fn get_native2 = NULL;
  npunlock_ze_graph_get_native_fn get_native = NULL;
  uint8_t *serialized = NULL;
  size_t serialized_size = 0;
  char *build_flags = NULL;
  uint32_t code;

  memset(result, 0, sizeof(*result));
  result->status = NPUNLOCK_IR_WORKER_DRIVER_FAILED;
  loader = LoadLibraryExW(L"ze_loader.dll", NULL, LOAD_LIBRARY_SEARCH_SYSTEM32);
  if (loader == NULL) {
    result->status = NPUNLOCK_IR_WORKER_LOADER_NOT_FOUND;
    set_diagnostic(result, "Level Zero loader ze_loader.dll was not found in System32");
    goto done;
  }
  if (!load_ze_functions(loader, &ze)) {
    result->status = NPUNLOCK_IR_WORKER_SYMBOL_MISSING;
    set_diagnostic(result, "Level Zero loader is missing a required export");
    goto done;
  }
  if (!select_npu(request, &ze, result, &driver, &device)) {
    goto done;
  }
  memset(&driver_properties, 0, sizeof(driver_properties));
  driver_properties.stype = NPUNLOCK_ZE_STRUCTURE_TYPE_DRIVER_PROPERTIES;
  code = ze.driver_get_properties(driver, &driver_properties);
  if (code != NPUNLOCK_ZE_SUCCESS) {
    set_driver_error(result, code, "zeDriverGetProperties");
    goto done;
  }
  result->driver_version = driver_properties.driverVersion;
  result->graph_version = find_graph_version(&ze, driver, result);
  if (result->graph_version == 0 ||
      !get_graph_ddi(&ze, driver, result->graph_version, result, &ddi) ||
      !query_graph_contract(ddi, device, result)) {
    goto done;
  }
  serialized = package_ir(request, result, &serialized_size);
  build_flags = (char *)malloc(request->build_flags_size + 1);
  if (serialized == NULL || build_flags == NULL) {
    result->status = NPUNLOCK_IR_WORKER_OUT_OF_MEMORY;
    goto done;
  }
  if (request->build_flags_size != 0) {
    memcpy(build_flags, request->build_flags, request->build_flags_size);
  }
  build_flags[request->build_flags_size] = '\0';
  memset(&context_desc, 0, sizeof(context_desc));
  context_desc.stype = NPUNLOCK_ZE_STRUCTURE_TYPE_CONTEXT_DESC;
  code = ze.context_create(driver, &context_desc, &context);
  if (code != NPUNLOCK_ZE_SUCCESS) {
    set_driver_error(result, code, "zeContextCreate");
    goto done;
  }
  memcpy(&create2, &ddi->slots[NPUNLOCK_ZE_GRAPH_DDI_CREATE2], sizeof(create2));
  memcpy(&destroy, &ddi->slots[NPUNLOCK_ZE_GRAPH_DDI_DESTROY], sizeof(destroy));
  memcpy(&get_native2, &ddi->slots[NPUNLOCK_ZE_GRAPH_DDI_GET_NATIVE2], sizeof(get_native2));
  memcpy(&get_native, &ddi->slots[NPUNLOCK_ZE_GRAPH_DDI_GET_NATIVE], sizeof(get_native));
  if (create2 == NULL || destroy == NULL || (get_native2 == NULL && get_native == NULL)) {
    result->status = NPUNLOCK_IR_WORKER_UNSUPPORTED;
    set_diagnostic(result, "required graph Create2/export entry point is unavailable");
    goto done;
  }
  memset(&graph_desc, 0, sizeof(graph_desc));
  graph_desc.stype = NPUNLOCK_ZE_STRUCTURE_TYPE_GRAPH_DESC_2;
  graph_desc.format = NPUNLOCK_ZE_GRAPH_FORMAT_NGRAPH_LITE;
  graph_desc.inputSize = serialized_size;
  graph_desc.pInput = serialized;
  graph_desc.pBuildFlags = build_flags;
  code = create2(context, device, &graph_desc, &graph);
  if (code != NPUNLOCK_ZE_SUCCESS || graph == NULL) {
    set_driver_error(result, code, "pfnCreate2(NGRAPH_LITE)");
    goto done;
  }
  if (get_native2 != NULL) {
    const uint8_t *view = NULL;
    size_t size = 0;
    code = get_native2(graph, &size, &view);
    if (code != NPUNLOCK_ZE_SUCCESS || view == NULL || size == 0 ||
        size > NPUNLOCK_IR_MAX_RESULT_SIZE) {
      set_driver_error(result, code, "pfnGetNativeBinary2");
      goto done;
    }
    result->blob = (uint8_t *)malloc(size);
    if (result->blob == NULL) {
      result->status = NPUNLOCK_IR_WORKER_OUT_OF_MEMORY;
      goto done;
    }
    memcpy(result->blob, view, size);
    result->blob_size = size;
  } else {
    size_t size = 0;
    code = get_native(graph, &size, NULL);
    if (code != NPUNLOCK_ZE_SUCCESS || size == 0 || size > NPUNLOCK_IR_MAX_RESULT_SIZE) {
      set_driver_error(result, code, "pfnGetNativeBinary(size)");
      goto done;
    }
    result->blob = (uint8_t *)malloc(size);
    if (result->blob == NULL) {
      result->status = NPUNLOCK_IR_WORKER_OUT_OF_MEMORY;
      goto done;
    }
    result->blob_size = size;
    code = get_native(graph, &size, result->blob);
    if (code != NPUNLOCK_ZE_SUCCESS || size > result->blob_size) {
      set_driver_error(result, code, "pfnGetNativeBinary(data)");
      goto done;
    }
    result->blob_size = size;
  }
  if (result->blob_size < 4 || result->blob[0] != 0x7f || result->blob[1] != 'E' ||
      result->blob[2] != 'L' || result->blob[3] != 'F') {
    result->status = NPUNLOCK_IR_WORKER_BAD_RESULT;
    set_diagnostic(result, "compiled native graph is not an ELF container");
    goto done;
  }
  code = destroy(graph);
  graph = NULL;
  if (code != NPUNLOCK_ZE_SUCCESS) {
    set_driver_error(result, code, "pfnDestroy(compiled graph)");
    goto done;
  }
  memset(&graph_desc, 0, sizeof(graph_desc));
  graph_desc.stype = NPUNLOCK_ZE_STRUCTURE_TYPE_GRAPH_DESC_2;
  graph_desc.format = NPUNLOCK_ZE_GRAPH_FORMAT_NATIVE;
  graph_desc.inputSize = result->blob_size;
  graph_desc.pInput = result->blob;
  graph_desc.pBuildFlags = NULL;
  code = create2(context, device, &graph_desc, &graph);
  if (code != NPUNLOCK_ZE_SUCCESS || graph == NULL) {
    set_driver_error(result, code, "pfnCreate2(native reload)");
    goto done;
  }
  result->status = NPUNLOCK_IR_WORKER_OK;
  result->driver_result = NPUNLOCK_ZE_SUCCESS;

done:
  if (graph != NULL && destroy != NULL) {
    code = destroy(graph);
    if (code != NPUNLOCK_ZE_SUCCESS && result->status == NPUNLOCK_IR_WORKER_OK) {
      set_driver_error(result, code, "pfnDestroy(graph)");
    }
  }
  if (context != NULL) {
    code = ze.context_destroy(context);
    if (code != NPUNLOCK_ZE_SUCCESS && result->status == NPUNLOCK_IR_WORKER_OK) {
      set_driver_error(result, code, "zeContextDestroy");
    }
  }
  if (result->status != NPUNLOCK_IR_WORKER_OK) {
    free(result->blob);
    result->blob = NULL;
    result->blob_size = 0;
  }
  free(build_flags);
  free(serialized);
  if (loader != NULL) {
    FreeLibrary(loader);
  }
}

static bool send_response(HANDLE handle, const ir_result *result) {
  uint8_t header[NPUNLOCK_IR_RESPONSE_HEADER_SIZE] = {0};
  size_t diagnostic_size = result->diagnostic == NULL ? 0 : strlen(result->diagnostic);
  store_u32(header, NPUNLOCK_IR_RESPONSE_MAGIC);
  store_u32(header + 4, NPUNLOCK_IR_PROTOCOL_VERSION);
  store_u32(header + 8, (uint32_t)result->status);
  store_u32(header + 12, result->driver_result);
  store_u32(header + 16, result->driver_index);
  store_u32(header + 20, result->device_index);
  store_u32(header + 24, result->graph_version);
  store_u32(header + 28,
            (uint32_t)result->compiler_major | ((uint32_t)result->compiler_minor << 16));
  store_u32(header + 32, result->max_opset);
  store_u32(header + 36, result->driver_version);
  store_u32(header + 40, result->vendor_id);
  store_u32(header + 44, result->device_id);
  store_u32(header + 48, result->elf_version.major);
  store_u32(header + 52, result->elf_version.minor);
  store_u32(header + 56, result->elf_version.patch);
  store_u32(header + 60, result->runtime_version.major);
  store_u32(header + 64, result->runtime_version.minor);
  store_u32(header + 68, result->runtime_version.patch);
  store_u64(header + 72, result->blob_size);
  store_u64(header + 80, diagnostic_size);
  return write_all(handle, header, sizeof(header)) &&
         write_all(handle, result->blob, result->blob_size) &&
         write_all(handle, (const uint8_t *)result->diagnostic, diagnostic_size);
}

int npunlock_ir_worker_main(int argc, char **argv) {
  HANDLE response = NULL;
  HANDLE input = GetStdHandle(STD_INPUT_HANDLE);
  uint8_t *request_data = NULL;
  size_t request_size = 0;
  ir_request request;
  ir_result result;
  int exit_code = 2;

  SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX | SEM_NOOPENFILEERRORBOX);
  if (!npunlock_worker_response_handle(argc, argv, &response)) {
    return 2;
  }
  memset(&result, 0, sizeof(result));
  result.status = NPUNLOCK_IR_WORKER_BAD_REQUEST;
  if (input == NULL || input == INVALID_HANDLE_VALUE ||
      !npunlock_worker_read_all(input, NPUNLOCK_IR_MAX_MESSAGE_SIZE, &request_data,
                                &request_size)) {
    set_diagnostic(&result, "failed to read the IR worker request");
  } else if (!parse_request(request_data, request_size, &request)) {
    set_diagnostic(&result, "malformed IR worker request");
  } else {
    compile_ir(&request, &result);
  }
  exit_code = send_response(response, &result) ? 0 : 3;
  free(result.blob);
  free(result.diagnostic);
  free(request_data);
  CloseHandle(response);
  return exit_code;
}
