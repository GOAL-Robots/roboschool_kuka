#!/bin/bash

set -e

SCRIPT=$(realpath -s $0)
SCRIPTPATH=$(dirname $SCRIPT)
SESSION=dmpgrasp
XORG_DISPLAY=:5
START=0
N_SIMS=4


# -------------------------------------------------------------------------------------------

# waits for process to end
# :paran $1: pid of the process
wait_proc() {
    pid=$1
    while [[ ! -z "$(ps -e| grep "^$1")" ]]; do sleep 0.1; done
}

# kills all current $SESSION sessions
kill_session() { 
    
    sessions=$(count_sessions)      
    
    echo "killing previous sessions..."
   
        
    for ses in $sessions;    
    do      
        echo "   killing session $ses" 
        screen -S "${SESSION}" -X quit &> /dev/null || true 
    done;
    screen -wipe &> /dev/null || true 
    echo "Done."
}

kill_xvfb() {
    killall -9 Xvfb &> /dev/null || true
} 


# init the screen session
init_screen() {

    kill_session
    sleep 3
    screen -dmS $SESSION
}

# :param $1: title of the window
prepare_window() {
    
    title=$1
    
    echo "Prepare $title in $SESSION:"

    screen -S $SESSION -X screen
    sleep 0.1
    screen -S $SESSION -X title $title 
    sleep 0.1
    screen -S $SESSION -X -p $title stuff "touch /tmp/${title}_prepared;\n"
    while [[ ! -f "/tmp/${title}_prepared" ]]; do echo "    Opening window ${title}..."; sleep 0.1; done
    rm -f /tmp/${title}_prepared
    echo "window ${title} opened."
    sleep 0.1
}

# :param $@: list of the windows to display in the current terminal
display_windows() {
    while [[ -z "$(screen -ls| grep $SESSION)" ]]; do sleep 0.1; done
    for w in "$@"; do
       screen -S $SESSION -X focus 
       screen -S $SESSION -X select $w 
       screen -S $SESSION -X split 
    done
    screen -S $SESSION -X focus 
    screen -S $SESSION -X remove 
}

# :param $1: title of the window
# :param $2: command to be executed
# :param $3: sleep interval after command (default 1)
exec_on_window() {
    title=$1
    comm=$2
    interval=${3:-1}
    screen -S $SESSION -p $title -X stuff "${comm} 2>&1 | tee ${title}_log \n"
    sleep $interval
}

# :param $1: title of the window
# :param $2: command to be executed
# :param $3: sleep interval after command (default 1)
exec_on_window_no_log() {
    title=$1
    comm=$2
    interval=${3:-1}
    screen -S $SESSION -p $title -X stuff "${comm} \n"
    sleep $interval
}

# return the list of session pids
count_sessions() {
    echo -n "$(screen -ls | \
            grep "$SESSION"| \
            grep -o '^\s\+[0-9]\+'|\
            sed 's/^\s\+\([0-9]\+\)[\.].*/\1/')" 
}

# -------------------------------------------------------------------------------------------


kill_xvfb
kill_session
