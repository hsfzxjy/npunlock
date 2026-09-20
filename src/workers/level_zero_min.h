#ifndef NPUNLOCK_LEVEL_ZERO_MIN_H
#define NPUNLOCK_LEVEL_ZERO_MIN_H

/*
 * Minimal private declarations for the stable Level Zero loader ABI and the
 * Intel NPU graph-extension prefix used by this project. Values and layouts
 * follow the MIT-licensed Level Zero v1.17 and Intel NPU extension v1.20
 * headers recorded in LICENSES/PROVENANCE.md. No public npunlock ABI exposes
 * these declarations.
 */

#include <stddef.h>
#include <stdint.h>

#define NPUNLOCK_ZE_SUCCESS 0
#define NPUNLOCK_ZE_STRUCTURE_TYPE_DRIVER_PROPERTIES 0x1
#define NPUNLOCK_ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES 0x3
#define NPUNLOCK_ZE_STRUCTURE_TYPE_CONTEXT_DESC 0xd
#define NPUNLOCK_ZE_STRUCTURE_TYPE_INIT_DRIVER_TYPE_DESC 0x00020021
#define NPUNLOCK_ZE_INIT_DRIVER_TYPE_FLAG_NPU 0x2
#define NPUNLOCK_ZE_DEVICE_TYPE_VPU 5
#define NPUNLOCK_ZE_STRUCTURE_TYPE_DRIVER_EXTENSION_NPU_EXT 0x1
#define NPUNLOCK_ZE_STRUCTURE_TYPE_DEVICE_GRAPH_PROPERTIES 0x1
#define NPUNLOCK_ZE_STRUCTURE_TYPE_DEVICE_GRAPH_PROPERTIES_2 0xf
#define NPUNLOCK_ZE_STRUCTURE_TYPE_GRAPH_DESC_2 0xe
#define NPUNLOCK_ZE_GRAPH_FORMAT_NATIVE 0x1
#define NPUNLOCK_ZE_GRAPH_FORMAT_NGRAPH_LITE 0x2
#define NPUNLOCK_ZE_GRAPH_DDI_CREATE2 16
#define NPUNLOCK_ZE_GRAPH_DDI_DESTROY 1
#define NPUNLOCK_ZE_GRAPH_DDI_GET_NATIVE 7
#define NPUNLOCK_ZE_GRAPH_DDI_DEVICE_PROPERTIES 8
#define NPUNLOCK_ZE_GRAPH_DDI_DEVICE_PROPERTIES2 19
#define NPUNLOCK_ZE_GRAPH_DDI_GET_NATIVE2 20

typedef struct npunlock_ze_driver_handle *npunlock_ze_driver_handle;
typedef struct npunlock_ze_device_handle *npunlock_ze_device_handle;
typedef struct npunlock_ze_context_handle *npunlock_ze_context_handle;
typedef struct npunlock_ze_graph_handle *npunlock_ze_graph_handle;

typedef struct npunlock_ze_init_driver_type_desc {
  uint32_t stype;
  const void *pNext;
  uint32_t flags;
} npunlock_ze_init_driver_type_desc;

typedef struct npunlock_ze_uuid {
  uint8_t id[16];
} npunlock_ze_uuid;

typedef struct npunlock_ze_driver_properties {
  uint32_t stype;
  void *pNext;
  npunlock_ze_uuid uuid;
  uint32_t driverVersion;
} npunlock_ze_driver_properties;

typedef struct npunlock_ze_driver_extension_properties {
  char name[256];
  uint32_t version;
} npunlock_ze_driver_extension_properties;

typedef struct npunlock_ze_device_properties {
  uint32_t stype;
  void *pNext;
  uint32_t type;
  uint32_t vendorId;
  uint32_t deviceId;
  uint32_t flags;
  uint32_t subdeviceId;
  uint32_t coreClockRate;
  uint64_t maxMemAllocSize;
  uint32_t maxHardwareContexts;
  uint32_t maxCommandQueuePriority;
  uint32_t numThreadsPerEU;
  uint32_t physicalEUSimdWidth;
  uint32_t numEUsPerSubslice;
  uint32_t numSubslicesPerSlice;
  uint32_t numSlices;
  uint64_t timerResolution;
  uint32_t timestampValidBits;
  uint32_t kernelTimestampValidBits;
  npunlock_ze_uuid uuid;
  char name[256];
} npunlock_ze_device_properties;

typedef struct npunlock_ze_context_desc {
  uint32_t stype;
  const void *pNext;
  uint32_t flags;
} npunlock_ze_context_desc;

typedef struct npunlock_ze_driver_extension_npu {
  uint32_t stype;
  const void *pNext;
  const char *name;
  uint32_t version;
  void **ppFunctionAddress;
} npunlock_ze_driver_extension_npu;

typedef struct npunlock_ze_graph_compiler_version {
  uint16_t major;
  uint16_t minor;
} npunlock_ze_graph_compiler_version;

typedef struct npunlock_ze_graph_version {
  uint32_t major;
  uint32_t minor;
  uint32_t patch;
} npunlock_ze_graph_version;

typedef struct npunlock_ze_device_graph_properties {
  uint32_t stype;
  void *pNext;
  uint32_t graphExtensionVersion;
  npunlock_ze_graph_compiler_version compilerVersion;
  uint32_t graphFormatsSupported;
  uint32_t maxOVOpsetVersionSupported;
} npunlock_ze_device_graph_properties;

typedef struct npunlock_ze_device_graph_properties_2 {
  uint32_t stype;
  void *pNext;
  uint32_t graphExtensionVersion;
  npunlock_ze_graph_compiler_version compilerVersion;
  uint32_t graphFormatsSupported;
  uint32_t maxOVOpsetVersionSupported;
  npunlock_ze_graph_version elfVersion;
  npunlock_ze_graph_version runtimeVersion;
} npunlock_ze_device_graph_properties_2;

typedef struct npunlock_ze_graph_desc_2 {
  uint32_t stype;
  void *pNext;
  uint32_t format;
  size_t inputSize;
  const uint8_t *pInput;
  const char *pBuildFlags;
  uint32_t flags;
} npunlock_ze_graph_desc_2;

typedef struct npunlock_ze_graph_ddi_prefix {
  void *slots[21];
} npunlock_ze_graph_ddi_prefix;

typedef uint32_t(__cdecl *npunlock_ze_init_drivers_fn)(uint32_t *, npunlock_ze_driver_handle *,
                                                       npunlock_ze_init_driver_type_desc *);
typedef uint32_t(__cdecl *npunlock_ze_driver_get_properties_fn)(npunlock_ze_driver_handle,
                                                                npunlock_ze_driver_properties *);
typedef uint32_t(__cdecl *npunlock_ze_driver_get_extensions_fn)(
    npunlock_ze_driver_handle, uint32_t *, npunlock_ze_driver_extension_properties *);
typedef uint32_t(__cdecl *npunlock_ze_driver_get_extension_address_fn)(npunlock_ze_driver_handle,
                                                                       const char *, void **);
typedef uint32_t(__cdecl *npunlock_ze_device_get_fn)(npunlock_ze_driver_handle, uint32_t *,
                                                     npunlock_ze_device_handle *);
typedef uint32_t(__cdecl *npunlock_ze_device_get_properties_fn)(npunlock_ze_device_handle,
                                                                npunlock_ze_device_properties *);
typedef uint32_t(__cdecl *npunlock_ze_context_create_fn)(npunlock_ze_driver_handle,
                                                         const npunlock_ze_context_desc *,
                                                         npunlock_ze_context_handle *);
typedef uint32_t(__cdecl *npunlock_ze_context_destroy_fn)(npunlock_ze_context_handle);
typedef uint32_t(__cdecl *npunlock_ze_npu_get_extension_fn)(npunlock_ze_driver_handle,
                                                            npunlock_ze_driver_extension_npu *);
typedef uint32_t(__cdecl *npunlock_ze_graph_device_properties_fn)(
    npunlock_ze_device_handle, npunlock_ze_device_graph_properties *);
typedef uint32_t(__cdecl *npunlock_ze_graph_device_properties2_fn)(
    npunlock_ze_device_handle, npunlock_ze_device_graph_properties_2 *);
typedef uint32_t(__cdecl *npunlock_ze_graph_create2_fn)(npunlock_ze_context_handle,
                                                        npunlock_ze_device_handle,
                                                        const npunlock_ze_graph_desc_2 *,
                                                        npunlock_ze_graph_handle *);
typedef uint32_t(__cdecl *npunlock_ze_graph_destroy_fn)(npunlock_ze_graph_handle);
typedef uint32_t(__cdecl *npunlock_ze_graph_get_native_fn)(npunlock_ze_graph_handle, size_t *,
                                                           uint8_t *);
typedef uint32_t(__cdecl *npunlock_ze_graph_get_native2_fn)(npunlock_ze_graph_handle, size_t *,
                                                            const uint8_t **);

#endif
