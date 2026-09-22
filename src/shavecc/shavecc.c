#include "npunlock/shavecc.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "elf32.h"
#include "internal.h"
#include "movi_stage.h"

static npunlock_view literal_view(const char *value) {
  return (npunlock_view){(const uint8_t *)value, strlen(value)};
}

static bool view_equals(npunlock_view view, const char *value) {
  size_t size = strlen(value);
  return view.size == size && memcmp(view.data, value, size) == 0;
}

static bool view_has_nul(npunlock_view view) {
  return view.size != 0 && memchr(view.data, 0, view.size) != NULL;
}

static bool is_absolute_windows_path(npunlock_view path) {
  if (path.size >= 3 &&
      ((path.data[0] >= 'A' && path.data[0] <= 'Z') ||
       (path.data[0] >= 'a' && path.data[0] <= 'z')) &&
      path.data[1] == ':' && (path.data[2] == '\\' || path.data[2] == '/')) {
    return true;
  }
  return path.size >= 2 && path.data[0] == '\\' && path.data[1] == '\\';
}

static bool is_entry_symbol(npunlock_view entry) {
  size_t index;
  if (entry.size == 0 || entry.size > 255 ||
      !((entry.data[0] >= 'A' && entry.data[0] <= 'Z') ||
        (entry.data[0] >= 'a' && entry.data[0] <= 'z') || entry.data[0] == '_')) {
    return false;
  }
  for (index = 1; index < entry.size; ++index) {
    uint8_t value = entry.data[index];
    if (!((value >= 'A' && value <= 'Z') || (value >= 'a' && value <= 'z') ||
          (value >= '0' && value <= '9') || value == '_')) {
      return false;
    }
  }
  return true;
}

static bool is_compiler_definition(npunlock_view definition) {
  size_t index = 0;
  bool have_digit = false;
  if (!npunlock_view_is_valid(definition) || definition.size < 3 || definition.size > 255 ||
      definition.data[0] < 'A' || definition.data[0] > 'Z') {
    return false;
  }
  while (index < definition.size && definition.data[index] != '=') {
    uint8_t value = definition.data[index];
    if (!((value >= 'A' && value <= 'Z') || (index != 0 && value >= '0' && value <= '9') ||
          value == '_')) {
      return false;
    }
    ++index;
  }
  if (index == 0 || index == definition.size || definition.data[index] != '=') {
    return false;
  }
  ++index;
  for (; index < definition.size; ++index) {
    if (definition.data[index] < '0' || definition.data[index] > '9') {
      return false;
    }
    have_digit = true;
  }
  return have_digit;
}

static npunlock_status make_path(npunlock_view directory, const char *filename,
                                 npunlock_buffer *path) {
  size_t filename_size = strlen(filename);
  bool add_separator =
      directory.data[directory.size - 1] != '\\' && directory.data[directory.size - 1] != '/';
  size_t size;
  uint8_t *data;
  if (!npunlock_checked_add_size(directory.size, add_separator ? 1 : 0, &size) ||
      !npunlock_checked_add_size(size, filename_size, &size)) {
    return NPUNLOCK_STATUS_OVERFLOW;
  }
  data = (uint8_t *)malloc(size);
  if (data == NULL) {
    return NPUNLOCK_STATUS_OUT_OF_MEMORY;
  }
  memcpy(data, directory.data, directory.size);
  if (add_separator) {
    data[directory.size] = '\\';
  }
  memcpy(data + directory.size + (add_separator ? 1 : 0), filename, filename_size);
  return npunlock_buffer_adopt_malloc(data, size, path);
}

static npunlock_status append_log(npunlock_buffer *destination, npunlock_view source) {
  size_t combined_size;
  uint8_t *combined;
  if (source.size == 0) {
    return NPUNLOCK_STATUS_OK;
  }
  if (!npunlock_checked_add_size(destination->size, source.size, &combined_size)) {
    return NPUNLOCK_STATUS_OVERFLOW;
  }
  combined = (uint8_t *)malloc(combined_size);
  if (combined == NULL) {
    return NPUNLOCK_STATUS_OUT_OF_MEMORY;
  }
  if (destination->size != 0) {
    memcpy(combined, destination->data, destination->size);
  }
  memcpy(combined + destination->size, source.data, source.size);
  npunlock_buffer_release(destination);
  return npunlock_buffer_adopt_malloc(combined, combined_size, destination);
}

static npunlock_status expand_npu3720_kernel_include(npunlock_view source,
                                                     npunlock_buffer *expanded,
                                                     npunlock_view *compiler_source) {
  static const char directive[] = "#include <npunlock/npu3720_kernel.h>";
  npunlock_view header = shavecc_npu3720_kernel_header();
  char line_marker[64];
  size_t offset = 0;
  size_t next_line = 1;
  size_t marker_size;
  size_t expanded_size;
  uint8_t *data;
  int marker_length;

  *compiler_source = source;
  while (offset < source.size && (source.data[offset] == ' ' || source.data[offset] == '\t' ||
                                  source.data[offset] == '\r' || source.data[offset] == '\n')) {
    if (source.data[offset] == '\n') {
      ++next_line;
    }
    ++offset;
  }
  if (source.size - offset < sizeof(directive) - 1 ||
      memcmp(source.data + offset, directive, sizeof(directive) - 1) != 0) {
    return NPUNLOCK_STATUS_OK;
  }
  offset += sizeof(directive) - 1;
  while (offset < source.size && (source.data[offset] == ' ' || source.data[offset] == '\t')) {
    ++offset;
  }
  if (offset < source.size && source.data[offset] == '\r') {
    ++offset;
    if (offset < source.size && source.data[offset] == '\n') {
      ++offset;
    }
    ++next_line;
  } else if (offset < source.size && source.data[offset] == '\n') {
    ++offset;
    ++next_line;
  } else if (offset != source.size) {
    return NPUNLOCK_STATUS_OK;
  }
  marker_length =
      snprintf(line_marker, sizeof(line_marker), "\n#line %zu \"entry.c\"\n", next_line);
  if (marker_length < 0 || (size_t)marker_length >= sizeof(line_marker)) {
    return NPUNLOCK_STATUS_INTERNAL_ERROR;
  }
  marker_size = (size_t)marker_length;
  if (!npunlock_checked_add_size(header.size, marker_size, &expanded_size) ||
      !npunlock_checked_add_size(expanded_size, source.size - offset, &expanded_size)) {
    return NPUNLOCK_STATUS_OVERFLOW;
  }
  data = (uint8_t *)malloc(expanded_size);
  if (data == NULL) {
    return NPUNLOCK_STATUS_OUT_OF_MEMORY;
  }
  memcpy(data, header.data, header.size);
  memcpy(data + header.size, line_marker, marker_size);
  memcpy(data + header.size + marker_size, source.data + offset, source.size - offset);
  if (npunlock_buffer_adopt_malloc(data, expanded_size, expanded) != NPUNLOCK_STATUS_OK) {
    free(data);
    return NPUNLOCK_STATUS_INTERNAL_ERROR;
  }
  *compiler_source = (npunlock_view){expanded->data, expanded->size};
  return NPUNLOCK_STATUS_OK;
}

static npunlock_status collect_worker_logs(shavecc_result *result,
                                           const npunlock_movi_result *worker) {
  npunlock_status status = append_log(
      &result->stdout_log, (npunlock_view){worker->stdout_log.data, worker->stdout_log.size});
  if (status == NPUNLOCK_STATUS_OK) {
    status = append_log(&result->stderr_log,
                        (npunlock_view){worker->stderr_log.data, worker->stderr_log.size});
  }
  return status;
}

static npunlock_status stage_failure(shavecc_result *result, npunlock_status status,
                                     const char *stage, const npunlock_movi_result *worker) {
  char fallback[160];
  char *message = NULL;
  size_t index;
  if (worker->diagnostic.size != 0 && worker->diagnostic.size <= 65535) {
    message = (char *)malloc(worker->diagnostic.size + 1);
    if (message != NULL) {
      memcpy(message, worker->diagnostic.data, worker->diagnostic.size);
      for (index = 0; index < worker->diagnostic.size; ++index) {
        if (message[index] == '\0') {
          message[index] = '?';
        }
      }
      message[worker->diagnostic.size] = '\0';
    }
  }
  if (message == NULL) {
    snprintf(fallback, sizeof(fallback),
             "Movi worker stage failed (worker_status=%u, tool_return=%d, process_exit=0x%08x)",
             worker->worker_status, worker->tool_return, worker->process_exit_code);
  }
  status = npunlock_set_diagnostic(&result->diagnostic, status, stage,
                                   message != NULL ? message : fallback);
  free(message);
  return status;
}

npunlock_status shavecc_compile(const shavecc_options *options, npunlock_view c_source,
                                shavecc_result *result) {
  enum { LINK_ENTRY_INDEX = 5, LINK_MATH_LIBRARY_INDEX = 9 };
  static const char *compile_base[] = {"moviCompile.dll",
                                       "-cc1",
                                       "-triple",
                                       "shave",
                                       "-target-cpu",
                                       NULL,
                                       "-S",
                                       "-O2",
                                       "-x",
                                       "c",
                                       "entry.c",
                                       "-ffunction-sections",
                                       "-fdata-sections"};
  static const char *compile_tail[] = {"-o", "-"};
  static const char *assemble_args[] = {"moviAsm.dll", "--cv", "3720xx", "--noSPrefixing"};
  static const char *link_base[] = {
      "moviLLD.dll",        "-flavor",       "gnu", "-EL", "-e", NULL, "-z",
      "max-page-size=0x10", "--gc-sections", NULL,
  };
  npunlock_view target;
  npunlock_view entry;
  npunlock_view linker_script;
  npunlock_view *compile_arguments = NULL;
  npunlock_view link_arguments[sizeof(link_base) / sizeof(link_base[0])];
  npunlock_view assemble_arguments[sizeof(assemble_args) / sizeof(assemble_args[0])];
  npunlock_view inputs[2];
  npunlock_buffer compiler_path = {0};
  npunlock_buffer assembler_path = {0};
  npunlock_buffer linker_path = {0};
  npunlock_buffer math_library_path = {0};
  npunlock_buffer expanded_source = {0};
  npunlock_buffer *paths[] = {&compiler_path, &assembler_path, &linker_path, &math_library_path};
  npunlock_movi_result compiled = {0};
  npunlock_movi_result assembled = {0};
  npunlock_movi_result linked = {0};
  npunlock_shave_image image;
  npunlock_status status = NPUNLOCK_STATUS_INVALID_ARGUMENT;
  size_t compile_count;
  size_t index;

  if (result == NULL) {
    return NPUNLOCK_STATUS_INVALID_ARGUMENT;
  }
  memset(result, 0, sizeof(*result));
  result->struct_size = (uint32_t)sizeof(*result);
  if (options == NULL || options->struct_size < sizeof(*options) ||
      !npunlock_view_is_valid(c_source) || c_source.size == 0 || view_has_nul(c_source) ||
      !npunlock_view_is_valid(options->movi_dll_directory_utf8) ||
      options->movi_dll_directory_utf8.size == 0 ||
      !is_absolute_windows_path(options->movi_dll_directory_utf8) ||
      view_has_nul(options->movi_dll_directory_utf8) ||
      !npunlock_view_is_valid(options->worker_executable_utf8) ||
      view_has_nul(options->worker_executable_utf8) ||
      !npunlock_view_is_valid(options->target_cpu) || view_has_nul(options->target_cpu) ||
      !npunlock_view_is_valid(options->entry_symbol) || view_has_nul(options->entry_symbol) ||
      (options->compiler_definition_count != 0 && options->compiler_definitions == NULL) ||
      options->compiler_definition_count > 64 || !npunlock_view_is_valid(options->linker_script) ||
      view_has_nul(options->linker_script) || options->timeout_ms == 0) {
    return npunlock_set_diagnostic(&result->diagnostic, NPUNLOCK_STATUS_INVALID_ARGUMENT,
                                   "shavecc.validate", "invalid options or C source view");
  }
  target = options->target_cpu.size == 0 ? literal_view("3720xx") : options->target_cpu;
  entry = options->entry_symbol.size == 0 ? literal_view("controlled_act") : options->entry_symbol;
  linker_script =
      options->linker_script.size == 0 ? shavecc_default_linker_script() : options->linker_script;
  if (!view_equals(target, "3720xx")) {
    return npunlock_set_diagnostic(&result->diagnostic, NPUNLOCK_STATUS_UNSUPPORTED,
                                   "shavecc.validate",
                                   "only the confirmed 3720xx target is supported");
  }
  if (!is_entry_symbol(entry)) {
    return npunlock_set_diagnostic(&result->diagnostic, NPUNLOCK_STATUS_INVALID_ARGUMENT,
                                   "shavecc.validate", "entry symbol is not a C identifier");
  }
  for (index = 0; index < options->compiler_definition_count; ++index) {
    if (!is_compiler_definition(options->compiler_definitions[index])) {
      return npunlock_set_diagnostic(
          &result->diagnostic, NPUNLOCK_STATUS_INVALID_ARGUMENT, "shavecc.validate",
          "compiler definitions must use the confirmed uppercase NAME=DECIMAL form");
    }
  }
  status = expand_npu3720_kernel_include(c_source, &expanded_source, &c_source);
  if (status != NPUNLOCK_STATUS_OK) {
    status = npunlock_set_diagnostic(&result->diagnostic, status, "shavecc.validate",
                                     "failed to expand the bundled NPU3720 kernel header");
    goto done;
  }
  status = make_path(options->movi_dll_directory_utf8, "bin\\moviCompile64.dll", &compiler_path);
  if (status == NPUNLOCK_STATUS_OK) {
    status = make_path(options->movi_dll_directory_utf8, "bin\\moviAsm64.dll", &assembler_path);
  }
  if (status == NPUNLOCK_STATUS_OK) {
    status = make_path(options->movi_dll_directory_utf8, "bin\\moviLLD64.dll", &linker_path);
  }
  if (status == NPUNLOCK_STATUS_OK) {
    status = make_path(options->movi_dll_directory_utf8, "lib\\mlibm.a", &math_library_path);
  }
  if (status != NPUNLOCK_STATUS_OK) {
    npunlock_set_diagnostic(&result->diagnostic, status, "shavecc.validate",
                            "failed to construct Movi DLL paths");
    goto done;
  }
  compile_count = sizeof(compile_base) / sizeof(compile_base[0]) +
                  options->compiler_definition_count +
                  sizeof(compile_tail) / sizeof(compile_tail[0]);
  compile_arguments = (npunlock_view *)calloc(compile_count, sizeof(*compile_arguments));
  if (compile_arguments == NULL) {
    status = npunlock_set_diagnostic(&result->diagnostic, NPUNLOCK_STATUS_OUT_OF_MEMORY,
                                     "shavecc.compile", "failed to allocate compiler arguments");
    goto done;
  }
  for (index = 0; index < sizeof(compile_base) / sizeof(compile_base[0]); ++index) {
    compile_arguments[index] =
        compile_base[index] == NULL ? target : literal_view(compile_base[index]);
  }
  for (index = 0; index < options->compiler_definition_count; ++index) {
    size_t argument_index = sizeof(compile_base) / sizeof(compile_base[0]) + index;
    size_t definition_size = options->compiler_definitions[index].size;
    uint8_t *definition = (uint8_t *)malloc(definition_size + 2);
    if (definition == NULL) {
      status = npunlock_set_diagnostic(&result->diagnostic, NPUNLOCK_STATUS_OUT_OF_MEMORY,
                                       "shavecc.compile", "failed to allocate compiler definition");
      goto done;
    }
    definition[0] = '-';
    definition[1] = 'D';
    memcpy(definition + 2, options->compiler_definitions[index].data, definition_size);
    compile_arguments[argument_index] = (npunlock_view){definition, definition_size + 2};
  }
  for (index = 0; index < sizeof(compile_tail) / sizeof(compile_tail[0]); ++index) {
    compile_arguments[compile_count - 2 + index] = literal_view(compile_tail[index]);
  }
  inputs[0] = c_source;
  status = npunlock_run_movi_stage(options->worker_executable_utf8,
                                   (npunlock_view){compiler_path.data, compiler_path.size},
                                   NPUNLOCK_MOVI_STAGE_COMPILE, compile_arguments, compile_count,
                                   inputs, 1, options->timeout_ms, &compiled);
  {
    npunlock_status log_status = collect_worker_logs(result, &compiled);
    if (log_status != NPUNLOCK_STATUS_OK) {
      status = npunlock_set_diagnostic(&result->diagnostic, log_status, "shavecc.compile",
                                       "failed to retain worker output");
      goto done;
    }
  }
  if (status != NPUNLOCK_STATUS_OK) {
    status = stage_failure(result, status, "shavecc.compile", &compiled);
    goto done;
  }
  for (index = 0; index < sizeof(assemble_args) / sizeof(assemble_args[0]); ++index) {
    assemble_arguments[index] = literal_view(assemble_args[index]);
  }
  inputs[0] = (npunlock_view){compiled.output.data, compiled.output.size};
  status = npunlock_run_movi_stage(options->worker_executable_utf8,
                                   (npunlock_view){assembler_path.data, assembler_path.size},
                                   NPUNLOCK_MOVI_STAGE_ASSEMBLE, assemble_arguments,
                                   sizeof(assemble_arguments) / sizeof(assemble_arguments[0]),
                                   inputs, 1, options->timeout_ms, &assembled);
  {
    npunlock_status log_status = collect_worker_logs(result, &assembled);
    if (log_status != NPUNLOCK_STATUS_OK) {
      status = npunlock_set_diagnostic(&result->diagnostic, log_status, "shavecc.assemble",
                                       "failed to retain worker output");
      goto done;
    }
  }
  if (status != NPUNLOCK_STATUS_OK) {
    status = stage_failure(result, status, "shavecc.assemble", &assembled);
    goto done;
  }
  for (index = 0; index < sizeof(link_base) / sizeof(link_base[0]); ++index) {
    if (link_base[index] != NULL) {
      link_arguments[index] = literal_view(link_base[index]);
    } else if (index == LINK_ENTRY_INDEX) {
      link_arguments[index] = entry;
    } else if (index == LINK_MATH_LIBRARY_INDEX) {
      link_arguments[index] = (npunlock_view){math_library_path.data, math_library_path.size};
    } else {
      status = npunlock_set_diagnostic(&result->diagnostic, NPUNLOCK_STATUS_INTERNAL_ERROR,
                                       "shavecc.link", "invalid linker argument template");
      goto done;
    }
  }
  inputs[0] = (npunlock_view){assembled.output.data, assembled.output.size};
  inputs[1] = linker_script;
  status = npunlock_run_movi_stage(
      options->worker_executable_utf8, (npunlock_view){linker_path.data, linker_path.size},
      NPUNLOCK_MOVI_STAGE_LINK, link_arguments, sizeof(link_arguments) / sizeof(link_arguments[0]),
      inputs, 2, options->timeout_ms, &linked);
  {
    npunlock_status log_status = collect_worker_logs(result, &linked);
    if (log_status != NPUNLOCK_STATUS_OK) {
      status = npunlock_set_diagnostic(&result->diagnostic, log_status, "shavecc.link",
                                       "failed to retain worker output");
      goto done;
    }
  }
  if (status != NPUNLOCK_STATUS_OK) {
    if (linked.diagnostic.size == 0) {
      char message[192];
      snprintf(message, sizeof(message),
               "Movi linker worker failed after %zu-byte assembly and %zu-byte object "
               "(worker_status=%u, tool_return=%d, process_exit=0x%08x)",
               compiled.output.size, assembled.output.size, linked.worker_status,
               linked.tool_return, linked.process_exit_code);
      status = npunlock_set_diagnostic(&result->diagnostic, status, "shavecc.link", message);
    } else {
      status = stage_failure(result, status, "shavecc.link", &linked);
    }
    goto done;
  }
  status = npunlock_parse_shave_elf((npunlock_view){linked.output.data, linked.output.size}, &image,
                                    &result->diagnostic);
  if (status != NPUNLOCK_STATUS_OK) {
    goto done;
  }
  result->elf = linked.output;
  memset(&linked.output, 0, sizeof(linked.output));

done:
  if (compile_arguments != NULL) {
    for (index = 0; index < options->compiler_definition_count; ++index) {
      size_t argument_index = sizeof(compile_base) / sizeof(compile_base[0]) + index;
      free((void *)compile_arguments[argument_index].data);
    }
  }
  free(compile_arguments);
  npunlock_buffer_release(&expanded_source);
  for (index = 0; index < sizeof(paths) / sizeof(paths[0]); ++index) {
    npunlock_buffer_release(paths[index]);
  }
  npunlock_movi_result_release(&compiled);
  npunlock_movi_result_release(&assembled);
  npunlock_movi_result_release(&linked);
  return status;
}

void shavecc_result_release(shavecc_result *result) {
  if (result == NULL) {
    return;
  }
  npunlock_buffer_release(&result->elf);
  npunlock_buffer_release(&result->stdout_log);
  npunlock_buffer_release(&result->stderr_log);
  npunlock_diagnostic_release(&result->diagnostic);
  memset(result, 0, sizeof(*result));
}
