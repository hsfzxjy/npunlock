#include "internal.h"

#include <string.h>

typedef struct sha256_context {
  uint32_t state[8];
  uint64_t bit_count;
  uint8_t block[64];
  size_t block_size;
} sha256_context;

static uint32_t rotate_right(uint32_t value, uint32_t count) {
  return (value >> count) | (value << (32u - count));
}

static uint32_t load_be32(const uint8_t *bytes) {
  return ((uint32_t)bytes[0] << 24) | ((uint32_t)bytes[1] << 16) | ((uint32_t)bytes[2] << 8) |
         (uint32_t)bytes[3];
}

static void store_be32(uint8_t *bytes, uint32_t value) {
  bytes[0] = (uint8_t)(value >> 24);
  bytes[1] = (uint8_t)(value >> 16);
  bytes[2] = (uint8_t)(value >> 8);
  bytes[3] = (uint8_t)value;
}

static void sha256_transform(sha256_context *context, const uint8_t block[64]) {
  static const uint32_t constants[64] = {
      0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u, 0x923f82a4u,
      0xab1c5ed5u, 0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu,
      0x9bdc06a7u, 0xc19bf174u, 0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu,
      0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau, 0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u,
      0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u, 0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu,
      0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u, 0xa2bfe8a1u, 0xa81a664bu,
      0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u, 0x19a4c116u,
      0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
      0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u, 0x90befffau, 0xa4506cebu, 0xbef9a3f7u,
      0xc67178f2u};
  uint32_t schedule[64];
  uint32_t a;
  uint32_t b;
  uint32_t c;
  uint32_t d;
  uint32_t e;
  uint32_t f;
  uint32_t g;
  uint32_t h;
  size_t index;

  for (index = 0; index < 16; ++index) {
    schedule[index] = load_be32(block + index * 4);
  }
  for (; index < 64; ++index) {
    uint32_t s0 = rotate_right(schedule[index - 15], 7) ^ rotate_right(schedule[index - 15], 18) ^
                  (schedule[index - 15] >> 3);
    uint32_t s1 = rotate_right(schedule[index - 2], 17) ^ rotate_right(schedule[index - 2], 19) ^
                  (schedule[index - 2] >> 10);
    schedule[index] = schedule[index - 16] + s0 + schedule[index - 7] + s1;
  }

  a = context->state[0];
  b = context->state[1];
  c = context->state[2];
  d = context->state[3];
  e = context->state[4];
  f = context->state[5];
  g = context->state[6];
  h = context->state[7];
  for (index = 0; index < 64; ++index) {
    uint32_t s1 = rotate_right(e, 6) ^ rotate_right(e, 11) ^ rotate_right(e, 25);
    uint32_t choose = (e & f) ^ ((~e) & g);
    uint32_t temporary1 = h + s1 + choose + constants[index] + schedule[index];
    uint32_t s0 = rotate_right(a, 2) ^ rotate_right(a, 13) ^ rotate_right(a, 22);
    uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
    uint32_t temporary2 = s0 + majority;
    h = g;
    g = f;
    f = e;
    e = d + temporary1;
    d = c;
    c = b;
    b = a;
    a = temporary1 + temporary2;
  }
  context->state[0] += a;
  context->state[1] += b;
  context->state[2] += c;
  context->state[3] += d;
  context->state[4] += e;
  context->state[5] += f;
  context->state[6] += g;
  context->state[7] += h;
}

static void sha256_update(sha256_context *context, const uint8_t *data, size_t size) {
  while (size != 0) {
    size_t available = sizeof(context->block) - context->block_size;
    size_t copied = size < available ? size : available;
    memcpy(context->block + context->block_size, data, copied);
    context->block_size += copied;
    data += copied;
    size -= copied;
    if (context->block_size == sizeof(context->block)) {
      sha256_transform(context, context->block);
      context->bit_count += 512;
      context->block_size = 0;
    }
  }
}

void npunlock_sha256(npunlock_view source, uint8_t digest[32]) {
  sha256_context context = {{0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au, 0x510e527fu,
                             0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u},
                            0,
                            {0},
                            0};
  uint64_t total_bits;
  size_t index;

  sha256_update(&context, source.data, source.size);
  total_bits = context.bit_count + (uint64_t)context.block_size * 8u;
  context.block[context.block_size++] = 0x80;
  if (context.block_size > 56) {
    memset(context.block + context.block_size, 0, sizeof(context.block) - context.block_size);
    sha256_transform(&context, context.block);
    context.block_size = 0;
  }
  memset(context.block + context.block_size, 0, 56 - context.block_size);
  for (index = 0; index < 8; ++index) {
    context.block[63 - index] = (uint8_t)(total_bits >> (index * 8));
  }
  sha256_transform(&context, context.block);
  for (index = 0; index < 8; ++index) {
    store_be32(digest + index * 4, context.state[index]);
  }
}
