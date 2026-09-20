#include "internal.h"

#include <stdint.h>

bool npunlock_checked_add_size(size_t left, size_t right, size_t *result) {
  if (result == NULL || right > SIZE_MAX - left) {
    return false;
  }
  *result = left + right;
  return true;
}

bool npunlock_checked_mul_size(size_t left, size_t right, size_t *result) {
  if (result == NULL || (left != 0 && right > SIZE_MAX / left)) {
    return false;
  }
  *result = left * right;
  return true;
}
