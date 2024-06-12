#include <iostream>
#include "version.h"

void printVersion() {
    std::cout << APP_NAME << " Version "
              << APP_VERSION_MAJOR << "."
              << APP_VERSION_MINOR << "."
              << APP_VERSION_PATCH << std::endl;
}