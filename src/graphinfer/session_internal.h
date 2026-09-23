#ifndef NPUNLOCK_GRAPHINFER_SESSION_INTERNAL_H
#define NPUNLOCK_GRAPHINFER_SESSION_INTERNAL_H

#include <stddef.h>

#include "npunlock/graphinfer.h"

npunlock_status npunlock_graphinfer_infer_copied(graphinfer_session *session,
                                                 const graphinfer_input *inputs, size_t input_count,
                                                 graphinfer_result *result);

#endif
