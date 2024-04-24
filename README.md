# README

Timing app and detector script to sync cameras together

Version: 0.0.1

## Licences ##

* external/GL: public domain
* external/glfw: see LICENCE.md file
* external/imgui: MIT licence
* external/stb: public domain

### The purpose ###

* run the timing app in the background of a video, then run the timing_detector.py script to detect the timestamps in each frame.
* enables synchronizing multiple cameras to each other without needing to connect all of them together.

### Installation

1. create a "build" folder under the root directory of this repository
2. run ''cmake -g Ninja ..''
3. run ''ninja''

