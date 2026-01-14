#include "print.hpp"

#include <sstream>

#include "exec.hpp"

int camspork_thread_local_print_program(
    size_t buffer_size, const void *program_buffer) {
  CAMSPORK_API_PROLOGUE
  std::stringstream s;
  camspork::print_program(
      s, buffer_size, static_cast<const char *>(program_buffer));
  camspork::thread_local_message_ref() = s.str();
  return 1;
  CAMSPORK_API_EPILOGUE(0)
}

int camspork_thread_local_print_program_with_remarks(
    camspork::ProgramEnv *p_env) {
  CAMSPORK_API_PROLOGUE
  std::stringstream s;
  p_env->stream_program_remarks(s);
  camspork::thread_local_message_ref() = s.str();
  return 1;
  CAMSPORK_API_EPILOGUE(0)
}
