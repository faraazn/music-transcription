# Copyright 2016 Google Inc. All Rights Reserved.
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

# Description:
# Libraries for using TensorFlow with music and art.

package(
    default_visibility = ["//amt:__subpackages__"],
)

licenses(["notice"])  # Apache 2.0

# The Magenta public API.
py_library(
    name = "music",
    srcs = ["__init__.py"],
    srcs_version = "PY2AND3",
    deps = [
    	":audio_io",
        ":midi_io",
        ":sequences_lib",
    ],
)

py_library(
    name = "audio_io",
    srcs = ["audio_io.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//amt/protobuf:music_py_pb2",
        # librosa dep
        # numpy dep
        # scipy dep
        # tensorflow dep
    ],
)

py_library(
    name = "midi_io",
    srcs = ["midi_io.py"],
    srcs_version = "PY2AND3",
    deps = [
	":constants",
        "//amt/protobuf:music_py_pb2",
        "@pretty_midi",
        # tensorflow dep
    ],
)

py_library(
    name = "sequences_lib",
    srcs = ["sequences_lib.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//amt/protobuf:music_py_pb2",
	":constants",
        # numpy dep
        # tensorflow dep
    ],
)

py_library(
    name = "constants",
    srcs = ["constants.py"],
    srcs_version = "PY2AND3",
)