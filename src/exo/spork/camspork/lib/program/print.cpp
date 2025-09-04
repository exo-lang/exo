#include "print.hpp"

#include <sstream>

int camspork_thread_local_print_program(size_t buffer_size, const void* program_buffer)
{
    CAMSPORK_API_PROLOGUE
    std::stringstream s;
    camspork::ProgramPrinter<std::stringstream>(s, buffer_size, static_cast<const char*>(program_buffer));
    camspork::thread_local_message_ref() = s.str();
    return 1;
    CAMSPORK_API_EPILOGUE(0)
}
