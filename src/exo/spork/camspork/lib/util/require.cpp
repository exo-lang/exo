#include "require.hpp"

namespace camspork
{

void require_fail(
    const char* file, int line, const char* expr_str, const char* msg)
{
    try {
        std::stringstream s;
        s << "Failed requirement: " << expr_str;
        s << " (" << msg << " @ " << file << ":" << line << ")";
        thread_local_message_ref() = s.str();
    }
    catch (...) {
        fprintf(stderr, "Exception while formatting error: %s:%i\n", __FILE__, __LINE__);
    }
    fprintf(stderr, "%s\n", thread_local_message_ref().c_str());
    throw RequireFail{};
}

std::string& thread_local_message_ref()
{
    thread_local std::string s;
    return s;
}

}

const char* camspork_thread_local_message_c_str()
{
    return ::camspork::thread_local_message_ref().c_str();
}
