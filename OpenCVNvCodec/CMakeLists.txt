#
# Copyright 2020 NVIDIA Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

cmake_minimum_required(VERSION 3.10)

project(OpenCVNvCodec)

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

message(STATUS "OpenCV include dirs ${OpenCV_INCLUDE_DIRS}" )

#Add src directory;
set (src_dir ${CMAKE_CURRENT_SOURCE_DIR}/src)
add_subdirectory(${src_dir})

#Do version stuff;
set (OPENCV_NVCODEC_VERSION_MAJOR 1)
set (OPENCV_NVCODEC_VERSION_MINOR 0)

#Set up python bindings;
set(GENERATE_OPENCV_EXTENSION FALSE CACHE BOOL "Generate VPF OpenCV extension")

if(GENERATE_PYTHON_BINDINGS)

	find_package(PythonLibs 3.5 REQUIRED)
	if(PYTHONLIBS_FOUND)

		#Add target;
		pybind11_add_module(OpenCVNvCodec SHARED ${PYOPENCVNVCODEC_SOURCES})
		set_property(TARGET OpenCVNvCodec PROPERTY CXX_STANDARD 14)

		#Link libs;
		target_link_libraries(OpenCVNvCodec PUBLIC ${PYTHON_LIBRARIES} ${NVCUVID_LIBRARY} ${NVENCODE_LIBRARY} ${OpenCV_LIBS})
	endif(PYTHONLIBS_FOUND)

endif(GENERATE_PYTHON_BINDINGS)

#Promote variables to parent scope;
set (GENERATE_OPENCV_EXTENSION	${GENERATE_OPENCV_EXTENSION} PARENT_SCOPE)
