#ifndef NPUNLOCK_API_H
#define NPUNLOCK_API_H

#if defined(_WIN32) && defined(NPUNLOCK_SHARED)
#define NPUNLOCK_EXPORT __declspec(dllexport)
#define NPUNLOCK_IMPORT __declspec(dllimport)
#elif defined(__GNUC__) && defined(NPUNLOCK_SHARED)
#define NPUNLOCK_EXPORT __attribute__((visibility("default")))
#define NPUNLOCK_IMPORT __attribute__((visibility("default")))
#else
#define NPUNLOCK_EXPORT
#define NPUNLOCK_IMPORT
#endif

#if defined(NPUNLOCK_COMMON_BUILDING)
#define NPUNLOCK_COMMON_API NPUNLOCK_EXPORT
#else
#define NPUNLOCK_COMMON_API NPUNLOCK_IMPORT
#endif

#if defined(NPUNLOCK_SHAVECC_BUILDING)
#define NPUNLOCK_SHAVECC_API NPUNLOCK_EXPORT
#else
#define NPUNLOCK_SHAVECC_API NPUNLOCK_IMPORT
#endif

#if defined(NPUNLOCK_IR2BLOB_BUILDING)
#define NPUNLOCK_IR2BLOB_API NPUNLOCK_EXPORT
#else
#define NPUNLOCK_IR2BLOB_API NPUNLOCK_IMPORT
#endif

#if defined(NPUNLOCK_PATCHBLOB_BUILDING)
#define NPUNLOCK_PATCHBLOB_API NPUNLOCK_EXPORT
#else
#define NPUNLOCK_PATCHBLOB_API NPUNLOCK_IMPORT
#endif

#endif
