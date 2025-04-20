#!/bin/bash

devices=$(seq 1 100)
# check the video device Available
found="none"
warn="[WARNING]: "
ok="[OK]: "
# considering a pc doesn't have more than 100 video devices
for i in $devices; do
    tmp="/dev/video$i"
    if [ -e $tmp ]; then
        found="/dev/video$i"
    fi
done

if [[ $found == "none" ]]; then
    echo "No device found"
fi
