#!/bin/bash
N=$1
cp ~/tmp/simulations/sim$N/frames/bests/rollout .
python test.py
convert -loop 0 -delay 1  $(find frames/lasts/| sort -n| tail -n 150) sim$N.gif


