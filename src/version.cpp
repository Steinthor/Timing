#include <iostream>
#include "version.h"

// Function to print version information
void printVersion() {
    std::cout << APP_NAME << " Version "
              << APP_VERSION_MAJOR << "."
              << APP_VERSION_MINOR << "."
              << APP_VERSION_PATCH << std::endl;
}

// Function to print help information
void printHelp() {
    std::cout << "./timing <argument>:\n"
              << "  -h, --help:  this helpful information.\n"
              << "  -v, --version: prints version of Timing app.\n"
              << "  <filepath>: saves a template image usable with the detector.\n";
}