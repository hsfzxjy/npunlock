#ifndef NPUNLOCK_COMMON_WORKER_PROTOCOL_H
#define NPUNLOCK_COMMON_WORKER_PROTOCOL_H

#include <stdint.h>

static inline uint32_t npunlock_worker_load_u32(const uint8_t *data) {
  return (uint32_t)data[0] | ((uint32_t)data[1] << 8) | ((uint32_t)data[2] << 16) |
         ((uint32_t)data[3] << 24);
}

static inline uint64_t npunlock_worker_load_u64(const uint8_t *data) {
  return (uint64_t)npunlock_worker_load_u32(data) |
         ((uint64_t)npunlock_worker_load_u32(data + 4) << 32);
}

static inline void npunlock_worker_store_u32(uint8_t *data, uint32_t value) {
  data[0] = (uint8_t)value;
  data[1] = (uint8_t)(value >> 8);
  data[2] = (uint8_t)(value >> 16);
  data[3] = (uint8_t)(value >> 24);
}

static inline void npunlock_worker_store_u64(uint8_t *data, uint64_t value) {
  npunlock_worker_store_u32(data, (uint32_t)value);
  npunlock_worker_store_u32(data + 4, (uint32_t)(value >> 32));
}

#endif
