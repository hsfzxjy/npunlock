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
#define NPUNLOCK_ZE_STRUCTURE_TYPE_COMMAND_QUEUE_GROUP_PROPERTIES 0x6
#define NPUNLOCK_ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC 0xe
#define NPUNLOCK_ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC 0xf
#define NPUNLOCK_ZE_STRUCTURE_TYPE_FENCE_DESC 0x12
#define NPUNLOCK_ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC 0x16
#define NPUNLOCK_ZE_STRUCTURE_TYPE_INIT_DRIVER_TYPE_DESC 0x00020021
#define NPUNLOCK_ZE_INIT_DRIVER_TYPE_FLAG_NPU 0x2
#define NPUNLOCK_ZE_DEVICE_TYPE_VPU 5
#define NPUNLOCK_ZE_STRUCTURE_TYPE_DRIVER_EXTENSION_NPU_EXT 0x1
#define NPUNLOCK_ZE_STRUCTURE_TYPE_DEVICE_GRAPH_PROPERTIES 0x1
#define NPUNLOCK_ZE_STRUCTURE_TYPE_DEVICE_GRAPH_PROPERTIES_2 0xf
#define NPUNLOCK_ZE_STRUCTURE_TYPE_GRAPH_DESC_2 0xe
#define NPUNLOCK_ZE_STRUCTURE_TYPE_GRAPH_PROPERTIES_3 0x11
#define NPUNLOCK_ZE_STRUCTURE_TYPE_GRAPH_ARGUMENT_PROPERTIES_3 0xd
#define NPUNLOCK_ZE_GRAPH_FORMAT_NATIVE 0x1
#define NPUNLOCK_ZE_GRAPH_FORMAT_NGRAPH_LITE 0x2
#define NPUNLOCK_ZE_GRAPH_DDI_CREATE2 16
#define NPUNLOCK_ZE_GRAPH_DDI_DESTROY 1
#define NPUNLOCK_ZE_GRAPH_DDI_GET_NATIVE 7
#define NPUNLOCK_ZE_GRAPH_DDI_DEVICE_PROPERTIES 8
#define NPUNLOCK_ZE_GRAPH_DDI_DEVICE_PROPERTIES2 19
#define NPUNLOCK_ZE_GRAPH_DDI_GET_NATIVE2 20
#define NPUNLOCK_ZE_GRAPH_DDI_GET_PROPERTIES3 26
#define NPUNLOCK_ZE_GRAPH_DDI_GET_ARGUMENT_PROPERTIES3 11
#define NPUNLOCK_ZE_GRAPH_DDI_SET_ARGUMENT 4
#define NPUNLOCK_ZE_GRAPH_DDI_APPEND_INITIALIZE 5
#define NPUNLOCK_ZE_GRAPH_DDI_APPEND_EXECUTE 6
#define NPUNLOCK_ZE_GRAPH_DDI_INITIALIZE 22
#define NPUNLOCK_ZE_GRAPH_STAGE_COMMAND_LIST_INITIALIZE 0x1
#define NPUNLOCK_ZE_GRAPH_STAGE_INITIALIZE 0x2
#define NPUNLOCK_ZE_GRAPH_ARGUMENT_TYPE_INPUT 0
#define NPUNLOCK_ZE_GRAPH_ARGUMENT_TYPE_OUTPUT 1
#define NPUNLOCK_ZE_GRAPH_ARGUMENT_PRECISION_FP16 0x2
#define NPUNLOCK_ZE_COMMAND_QUEUE_GROUP_COMPUTE 0x1
#define NPUNLOCK_ZE_HOST_MEM_ALLOC_WRITE_COMBINED 0x4

typedef struct npunlock_ze_driver_handle *npunlock_ze_driver_handle;
typedef struct npunlock_ze_device_handle *npunlock_ze_device_handle;
typedef struct npunlock_ze_context_handle *npunlock_ze_context_handle;
typedef struct npunlock_ze_graph_handle *npunlock_ze_graph_handle;
typedef struct npunlock_ze_command_queue_handle *npunlock_ze_command_queue_handle;
typedef struct npunlock_ze_command_list_handle *npunlock_ze_command_list_handle;
typedef struct npunlock_ze_fence_handle *npunlock_ze_fence_handle;
typedef struct npunlock_ze_event_handle *npunlock_ze_event_handle;
typedef struct npunlock_ze_graph_profiling_query_handle *npunlock_ze_graph_profiling_query_handle;

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

typedef struct npunlock_ze_graph_properties_3 {
  uint32_t stype;
  void *pNext;
  uint32_t numGraphArgs;
  uint32_t initStageRequired;
  uint32_t flags;
} npunlock_ze_graph_properties_3;

typedef struct npunlock_ze_graph_argument_properties_3 {
  uint32_t stype;
  void *pNext;
  char name[256];
  uint32_t type;
  uint32_t dims[5];
  uint32_t networkPrecision;
  uint32_t networkLayout;
  uint32_t devicePrecision;
  uint32_t deviceLayout;
  float quantReverseScale;
  uint8_t quantZeroPoint;
  uint32_t dims_count;
  char debug_friendly_name[256];
  char associated_tensor_names[32][256];
  uint32_t associated_tensor_names_count;
} npunlock_ze_graph_argument_properties_3;

typedef struct npunlock_ze_command_queue_group_properties {
  uint32_t stype;
  void *pNext;
  uint32_t flags;
  size_t maxMemoryFillPatternSize;
  uint32_t numQueues;
} npunlock_ze_command_queue_group_properties;

typedef struct npunlock_ze_command_queue_desc {
  uint32_t stype;
  const void *pNext;
  uint32_t ordinal;
  uint32_t index;
  uint32_t flags;
  uint32_t mode;
  uint32_t priority;
} npunlock_ze_command_queue_desc;

typedef struct npunlock_ze_command_list_desc {
  uint32_t stype;
  const void *pNext;
  uint32_t commandQueueGroupOrdinal;
  uint32_t flags;
} npunlock_ze_command_list_desc;

typedef struct npunlock_ze_fence_desc {
  uint32_t stype;
  const void *pNext;
  uint32_t flags;
} npunlock_ze_fence_desc;

typedef struct npunlock_ze_host_mem_alloc_desc {
  uint32_t stype;
  const void *pNext;
  uint32_t flags;
} npunlock_ze_host_mem_alloc_desc;

typedef struct npunlock_ze_graph_ddi_prefix {
  void *slots[30];
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
typedef uint32_t(__cdecl *npunlock_ze_device_get_queue_groups_fn)(
    npunlock_ze_device_handle, uint32_t *, npunlock_ze_command_queue_group_properties *);
typedef uint32_t(__cdecl *npunlock_ze_mem_alloc_host_fn)(npunlock_ze_context_handle,
                                                         const npunlock_ze_host_mem_alloc_desc *,
                                                         size_t, size_t, void **);
typedef uint32_t(__cdecl *npunlock_ze_mem_free_fn)(npunlock_ze_context_handle, void *);
typedef uint32_t(__cdecl *npunlock_ze_command_queue_create_fn)(
    npunlock_ze_context_handle, npunlock_ze_device_handle, const npunlock_ze_command_queue_desc *,
    npunlock_ze_command_queue_handle *);
typedef uint32_t(__cdecl *npunlock_ze_command_queue_destroy_fn)(npunlock_ze_command_queue_handle);
typedef uint32_t(__cdecl *npunlock_ze_command_queue_execute_fn)(npunlock_ze_command_queue_handle,
                                                                uint32_t,
                                                                npunlock_ze_command_list_handle *,
                                                                npunlock_ze_fence_handle);
typedef uint32_t(__cdecl *npunlock_ze_command_list_create_fn)(npunlock_ze_context_handle,
                                                              npunlock_ze_device_handle,
                                                              const npunlock_ze_command_list_desc *,
                                                              npunlock_ze_command_list_handle *);
typedef uint32_t(__cdecl *npunlock_ze_command_list_close_fn)(npunlock_ze_command_list_handle);
typedef uint32_t(__cdecl *npunlock_ze_command_list_destroy_fn)(npunlock_ze_command_list_handle);
typedef uint32_t(__cdecl *npunlock_ze_fence_create_fn)(npunlock_ze_command_queue_handle,
                                                       const npunlock_ze_fence_desc *,
                                                       npunlock_ze_fence_handle *);
typedef uint32_t(__cdecl *npunlock_ze_fence_destroy_fn)(npunlock_ze_fence_handle);
typedef uint32_t(__cdecl *npunlock_ze_fence_synchronize_fn)(npunlock_ze_fence_handle, uint64_t);
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
typedef uint32_t(__cdecl *npunlock_ze_graph_get_properties3_fn)(npunlock_ze_graph_handle,
                                                                npunlock_ze_graph_properties_3 *);
typedef uint32_t(__cdecl *npunlock_ze_graph_get_argument_properties3_fn)(
    npunlock_ze_graph_handle, uint32_t, npunlock_ze_graph_argument_properties_3 *);
typedef uint32_t(__cdecl *npunlock_ze_graph_set_argument_fn)(npunlock_ze_graph_handle, uint32_t,
                                                             const void *);
typedef uint32_t(__cdecl *npunlock_ze_graph_initialize_fn)(npunlock_ze_graph_handle);
typedef uint32_t(__cdecl *npunlock_ze_graph_append_initialize_fn)(npunlock_ze_command_list_handle,
                                                                  npunlock_ze_graph_handle,
                                                                  npunlock_ze_event_handle,
                                                                  uint32_t,
                                                                  npunlock_ze_event_handle *);
typedef uint32_t(__cdecl *npunlock_ze_graph_append_execute_fn)(
    npunlock_ze_command_list_handle, npunlock_ze_graph_handle,
    npunlock_ze_graph_profiling_query_handle, npunlock_ze_event_handle, uint32_t,
    npunlock_ze_event_handle *);

#endif
