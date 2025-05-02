file(GLOB SRCS src/*.cc)
include_directories(include)
add_library(hostop SHARED ${SRCS})
