# README

Timing app and detector script to sync cameras together

Version: 0.0.1

Tested on Ubuntu and WSL Ubuntu.  For Vsync to work through WSL you need to run it in full screen.

## Status

[![Cmake compilation](https://github.com/Steinthor/Timing/actions/workflows/github-actions.yml/badge.svg)](https://github.com/Steinthor/Timing/actions/workflows/github-actions.yml)

## Licences ##

* external/GL: public domain
* external/glfw: see LICENCE.md file
* external/imgui: MIT licence
* external/stb: public domain

### The purpose ###

* run the timing app in the background of a video, then run the timing_detector.py script to detect the timestamps in each frame.
* enables synchronizing multiple cameras to each other without needing to connect all of them together.

### Installation

1. run "build.sh" script to build and install the app
