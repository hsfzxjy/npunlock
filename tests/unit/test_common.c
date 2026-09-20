#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "internal.h"
#include "npunlock/buffer.h"
#include "npunlock/error.h"

#define CHECK(expression)                                                                          \
  do {                                                                                             \
    if (!(expression)) {                                                                           \
      fprintf(stderr, "check failed at %s:%d: %s\n", __FILE__, __LINE__, #expression);             \
      return 1;                                                                                    \
    }                                                                                              \
  } while (0)

int main(void) {
  const uint8_t bytes[] = {1, 2, 3, 4};
  npunlock_buffer copy = {0};
  npunlock_diagnostic diagnostic = {0};
  size_t result = 0;
  uint8_t digest[32];
  static const uint8_t abc_digest[32] = {0xba, 0x78, 0x16, 0xbf, 0x8f, 0x01, 0xcf, 0xea,
                                         0x41, 0x41, 0x40, 0xde, 0x5d, 0xae, 0x22, 0x23,
                                         0xb0, 0x03, 0x61, 0xa3, 0x96, 0x17, 0x7a, 0x9c,
                                         0xb4, 0x10, 0xff, 0x61, 0xf2, 0x00, 0x15, 0xad};

  CHECK(npunlock_view_is_valid((npunlock_view){NULL, 0}));
  CHECK(!npunlock_view_is_valid((npunlock_view){NULL, 1}));
  CHECK(npunlock_buffer_copy((npunlock_view){bytes, sizeof(bytes)}, &copy) == NPUNLOCK_STATUS_OK);
  CHECK(copy.size == sizeof(bytes));
  CHECK(copy.data != bytes);
  CHECK(memcmp(copy.data, bytes, sizeof(bytes)) == 0);
  npunlock_buffer_release(&copy);
  npunlock_buffer_release(&copy);

  CHECK(npunlock_checked_add_size(4, 5, &result) && result == 9);
  CHECK(!npunlock_checked_add_size(SIZE_MAX, 1, &result));
  CHECK(npunlock_checked_mul_size(4, 5, &result) && result == 20);
  CHECK(!npunlock_checked_mul_size(SIZE_MAX, 2, &result));

  npunlock_sha256((npunlock_view){(const uint8_t *)"abc", 3}, digest);
  CHECK(memcmp(digest, abc_digest, sizeof(digest)) == 0);

  CHECK(npunlock_set_diagnostic(&diagnostic, NPUNLOCK_STATUS_INVALID_ARGUMENT, "unit\"stage",
                                "line\nbreak") == NPUNLOCK_STATUS_INVALID_ARGUMENT);
  CHECK(diagnostic.json.data != NULL);
  CHECK(strstr((const char *)diagnostic.json.data, "invalid_argument") != NULL);
  CHECK(strstr((const char *)diagnostic.json.data, "unit\\\"stage") != NULL);
  CHECK(strstr((const char *)diagnostic.json.data, "line\\u000abreak") != NULL);
  npunlock_diagnostic_release(&diagnostic);
  npunlock_diagnostic_release(&diagnostic);

  CHECK(strcmp(npunlock_status_name(NPUNLOCK_STATUS_TIMEOUT), "timeout") == 0);
  CHECK(strcmp(npunlock_status_name((npunlock_status)999), "unknown") == 0);
  return 0;
}
