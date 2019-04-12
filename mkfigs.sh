#!/bin/bash
N=$1
#cp ~/tmp/simulations/sim$N/frames/epochs/rollout .
#python test.py
convert -loop 0 -delay 1  $(find frames/lasts/| sort -n| tail -n 150|awk 'NR%3==0') sim$N.gif


