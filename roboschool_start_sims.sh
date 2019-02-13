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
        screen -S "${SESSION}" -X quit &> /dev/null   
    done;
    killall -9 Xvfb &> /dev/null || true
    screen -wipe &> /dev/null || true 
    echo "Done."
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
    
    screen -S $SESSION -X screen
    sleep 0.1
    screen -S $SESSION -X title $title 
    sleep 0.1
    screen -S $SESSION -X -p $title stuff "touch /tmp/${title}_prepared;\n"
    while [[ ! -f "/tmp/${title}_prepared" ]]; do echo "Opening window for ${title}..."; sleep 0.1; done
    rm -f /tmp/${title}_prepared
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


run_script() {
    sim=$1
    echo "Starting $sim"
    
    exec_on_window_no_log $sim "cd ${HOME}/tmp"
    sleep 0.01
    exec_on_window_no_log $sim "mkdir -p simulations/$sim"
    sleep 0.01
    exec_on_window_no_log $sim "rm -fr simulations/$sim/*"
    sleep 0.01
    exec_on_window_no_log $sim "cd simulations/$sim"
    sleep 0.01
    exec_on_window_no_log $sim "DISPLAY=$XORG_DISPLAY vglrun python $SCRIPTPATH/roboschool_kuka_dmpgrasp.py"
    #exec_on_window_no_log $sim "python $SCRIPTPATH/roboschool_kuka_dmpgrasp.py"
    sleep 0.1
}




# prepare screen session
init_screen
echo ----

prepare_window xvfb

for x in $(seq $START $((START + N_SIMS - 1))); do
    prepare_window sim${x}
done

while [ "$(count_sessions)" -lt $N_SIMS ]; do sleep 0.1;  done     

exec_on_window_no_log xvfb "Xvfb $XORG_DISPLAY -screen 0 800x600x24"

for x in $(seq $START $((START + N_SIMS - 1))); do
    run_script sim${x}
done
