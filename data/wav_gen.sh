#!/bin/bash

IFS=$'\n'; set -f
for f in $(find clean_midi/ -name '*.mid' -or -name '*.midi'); do timidity -Ow --preserve-silence "$f"; done
unset IFS; set +f
