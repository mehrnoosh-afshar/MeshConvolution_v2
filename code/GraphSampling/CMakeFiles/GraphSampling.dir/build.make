# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.5

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /mnt/hdd1/yi_hdd1/GraphCNN_Facebook/public/code/GraphSampling

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /mnt/hdd1/yi_hdd1/GraphCNN_Facebook/public/code/GraphSampling

# Include any dependencies generated for this target.
include CMakeFiles/GraphSampling.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/GraphSampling.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/GraphSampling.dir/flags.make

CMakeFiles/GraphSampling.dir/main.cpp.o: CMakeFiles/GraphSampling.dir/flags.make
CMakeFiles/GraphSampling.dir/main.cpp.o: main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mnt/hdd1/yi_hdd1/GraphCNN_Facebook/public/code/GraphSampling/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/GraphSampling.dir/main.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/GraphSampling.dir/main.cpp.o -c /mnt/hdd1/yi_hdd1/GraphCNN_Facebook/public/code/GraphSampling/main.cpp

CMakeFiles/GraphSampling.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/GraphSampling.dir/main.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /mnt/hdd1/yi_hdd1/GraphCNN_Facebook/public/code/GraphSampling/main.cpp > CMakeFiles/GraphSampling.dir/main.cpp.i

CMakeFiles/GraphSampling.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/GraphSampling.dir/main.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /mnt/hdd1/yi_hdd1/GraphCNN_Facebook/public/code/GraphSampling/main.cpp -o CMakeFiles/GraphSampling.dir/main.cpp.s

CMakeFiles/GraphSampling.dir/main.cpp.o.requires:

.PHONY : CMakeFiles/GraphSampling.dir/main.cpp.o.requires

CMakeFiles/GraphSampling.dir/main.cpp.o.provides: CMakeFiles/GraphSampling.dir/main.cpp.o.requires
	$(MAKE) -f CMakeFiles/GraphSampling.dir/build.make CMakeFiles/GraphSampling.dir/main.cpp.o.provides.build
.PHONY : CMakeFiles/GraphSampling.dir/main.cpp.o.provides

CMakeFiles/GraphSampling.dir/main.cpp.o.provides.build: CMakeFiles/GraphSampling.dir/main.cpp.o


CMakeFiles/GraphSampling.dir/cnpy/cnpy.cpp.o: CMakeFiles/GraphSampling.dir/flags.make
CMakeFiles/GraphSampling.dir/cnpy/cnpy.cpp.o: cnpy/cnpy.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mnt/hdd1/yi_hdd1/GraphCNN_Facebook/public/code/GraphSampling/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/GraphSampling.dir/cnpy/cnpy.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/GraphSampling.dir/cnpy/cnpy.cpp.o -c /mnt/hdd1/yi_hdd1/GraphCNN_Facebook/public/code/GraphSampling/cnpy/cnpy.cpp

CMakeFiles/GraphSampling.dir/cnpy/cnpy.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/GraphSampling.dir/cnpy/cnpy.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /mnt/hdd1/yi_hdd1/GraphCNN_Facebook/public/code/GraphSampling/cnpy/cnpy.cpp > CMakeFiles/GraphSampling.dir/cnpy/cnpy.cpp.i

CMakeFiles/GraphSampling.dir/cnpy/cnpy.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/GraphSampling.dir/cnpy/cnpy.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /mnt/hdd1/yi_hdd1/GraphCNN_Facebook/public/code/GraphSampling/cnpy/cnpy.cpp -o CMakeFiles/GraphSampling.dir/cnpy/cnpy.cpp.s

CMakeFiles/GraphSampling.dir/cnpy/cnpy.cpp.o.requires:

.PHONY : CMakeFiles/GraphSampling.dir/cnpy/cnpy.cpp.o.requires

CMakeFiles/GraphSampling.dir/cnpy/cnpy.cpp.o.provides: CMakeFiles/GraphSampling.dir/cnpy/cnpy.cpp.o.requires
	$(MAKE) -f CMakeFiles/GraphSampling.dir/build.make CMakeFiles/GraphSampling.dir/cnpy/cnpy.cpp.o.provides.build
.PHONY : CMakeFiles/GraphSampling.dir/cnpy/cnpy.cpp.o.provides

CMakeFiles/GraphSampling.dir/cnpy/cnpy.cpp.o.provides.build: CMakeFiles/GraphSampling.dir/cnpy/cnpy.cpp.o


# Object files for target GraphSampling
GraphSampling_OBJECTS = \
"CMakeFiles/GraphSampling.dir/main.cpp.o" \
"CMakeFiles/GraphSampling.dir/cnpy/cnpy.cpp.o"

# External object files for target GraphSampling
GraphSampling_EXTERNAL_OBJECTS =

GraphSampling: CMakeFiles/GraphSampling.dir/main.cpp.o
GraphSampling: CMakeFiles/GraphSampling.dir/cnpy/cnpy.cpp.o
GraphSampling: CMakeFiles/GraphSampling.dir/build.make
GraphSampling: /home/zhouyi/anaconda3/lib/libz.so
GraphSampling: CMakeFiles/GraphSampling.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/mnt/hdd1/yi_hdd1/GraphCNN_Facebook/public/code/GraphSampling/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable GraphSampling"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/GraphSampling.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/GraphSampling.dir/build: GraphSampling

.PHONY : CMakeFiles/GraphSampling.dir/build

CMakeFiles/GraphSampling.dir/requires: CMakeFiles/GraphSampling.dir/main.cpp.o.requires
CMakeFiles/GraphSampling.dir/requires: CMakeFiles/GraphSampling.dir/cnpy/cnpy.cpp.o.requires

.PHONY : CMakeFiles/GraphSampling.dir/requires

CMakeFiles/GraphSampling.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/GraphSampling.dir/cmake_clean.cmake
.PHONY : CMakeFiles/GraphSampling.dir/clean

CMakeFiles/GraphSampling.dir/depend:
	cd /mnt/hdd1/yi_hdd1/GraphCNN_Facebook/public/code/GraphSampling && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /mnt/hdd1/yi_hdd1/GraphCNN_Facebook/public/code/GraphSampling /mnt/hdd1/yi_hdd1/GraphCNN_Facebook/public/code/GraphSampling /mnt/hdd1/yi_hdd1/GraphCNN_Facebook/public/code/GraphSampling /mnt/hdd1/yi_hdd1/GraphCNN_Facebook/public/code/GraphSampling /mnt/hdd1/yi_hdd1/GraphCNN_Facebook/public/code/GraphSampling/CMakeFiles/GraphSampling.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/GraphSampling.dir/depend

