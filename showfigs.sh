#!/bin/bash
N=$1
DIR=${2:-bests}
rm -r frames
cp ~/tmp/simulations/sim$N/frames/$DIR/rollout .
python test.py
# convert -loop 0 -delay 1  $(find frames/lasts/| \
#     sort -n| tail -n 200) sim$N.gif


