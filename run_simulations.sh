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
   
        
    screen -S "video" -X quit &> /dev/null || true  
    killall -9 physics_server > /dev/null 2>&1 || true
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

# html visualization template

html_figs()
{
    html='
    <!DOCTYPE html>
    <html>
    <head>
    <title></title>
    </head>
    <body>
    <table summary="">
    <col width="70%">
    <col width="30%">
    __TR__
    </table>
    </body>
    </html>'
    row='
    <tr>
    <td><img src="sim__NUM___b.gif" width="100%" alt=""></td>
    <td><img src="sim__NUM___rew.png" width="100%" alt=""></td>
    </tr>'

    for r in $( seq $START $((N_SIMS - 1)) ); do
        curr_row="$(echo $row | sed -e"s@__NUM__@$r@g")"
        html=$(echo "$html" | sed -e"s@__TR__@${curr_row}__TR__@g" )
    done
    html=$(echo "$html" | sed -e"s@__TR__@@g" )
    if [[ ! -z $(which tidy) ]]; then
        html="$(echo "$html"|tidy -miq || true)" 
    fi
    echo "$html"
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
    exec_on_window_no_log $sim "export DISPLAY=$XORG_DISPLAY"
    sleep 0.01 
    exec_on_window_no_log $sim "vglrun python $SCRIPTPATH/simulation.py 2>&1 | stdbuf -o0 grep -v kuka_gripper| tee log "
    sleep 0.1
}


# Manage arguments
# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------

usage()
{
    cat <<EOF

    usage: $0 options

    This script runs a group of kuka_gipper simulations 

    OPTIONS:
    -r --run         Starts the simulation
    -k --close       Closes all processes
    -l --last        Show last rollouts
    -h --help        Show this help menu 
EOF
}


RUN=false
CLOSE=false
BESTS=true
B3SERV=
VGL=

# getopt
GOTEMP="$(getopt -o "rklh" -l "run,close,last,help"  -n '' -- "$@")"

if [[ -z "$(echo -n $GOTEMP |sed -e"s/\-\-\(\s\+.*\|\s*\)$//")" ]]; then
    usage; exit;
fi

eval set -- "$GOTEMP"

while true ;
do
    case "$1" in
        -k | --close)
            CLOSE=true
            break ;;
        -r | --run)
            RUN=true
            shift;;
        -l | --last)
            BESTS=false
            shift;;
        -h | --help)
            echo "on help"
            usage; exit;
            shift;
            break;;
        --) shift ;
            break ;;
    esac
done

# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------


if [[ ${CLOSE} == true ]]; then
    kill_xvfb
    kill_session
fi

if [[ ${RUN} == true ]]; then
    # prepare screen session
    kill_xvfb
    init_screen
    echo ----
    sleep 0.1
    prepare_window xvfb

    for x in $(seq $START $((START + N_SIMS - 1))); do
        prepare_window sim${x}
    done

    sleep 0.1
    exec_on_window_no_log xvfb "Xvfb $XORG_DISPLAY -screen 0 800x600x24 &"
    sleep 1

    for x in $(seq $START $((START + N_SIMS - 1))); do
        run_script sim${x}
        sleep 5
    done

    sleep 2

    SESSION=video

    # prepare screen session
    init_screen 1
    echo ----
    sleep 0.1

    prepare_window mkvideo
    prepare_window jupyter

    sleep 0.1

    if [[ $BESTS == true ]]; then
        frame_dir=bests
    else
        frame_dir=lasts
    fi

    exec_on_window_no_log mkvideo "mkdir -p ${HOME}/tmp/simulations "
    exec_on_window_no_log mkvideo "cd ${HOME}/tmp/simulations"
    exec_on_window_no_log mkvideo "rm *gif *png"
    echo "$(html_figs)" > ${HOME}/tmp/simulations/figs.html
    exec_on_window_no_log mkvideo '
    echo "clear initial figures..."
    dirs=\$(find -type d| grep "sim[0-9]\\+\$") 
    for d in \$dirs; do
        convert -size 800x600 xc:transparent \${d}_b.gif
        convert -size 800x600 xc:transparent \${d}_l.gif
        dn=\$(basename \$d)
        dnr=\${dn}_rew.png
        convert -size 800x600 xc:transparent \$dnr
    done
    convert -size 800x600 xc:transparent rews.png
    '
    exec_on_window_no_log mkvideo '
    while [[ true ]]; do
        for d in \$dirs; do
            echo "make \${d}_rew.png ..."
            cp \$d/frames/rew.png \${d}_rew.png
        done
        convert +append *rew.png rews_tmp.png
        mv rews_tmp.png rews.png
        for d in \$dirs; do
            echo "make \${d}_b_tmp.gif ..."
            convert -loop 0 -delay 2 \$(find \$d/frames/$frame_dir/ | grep jpeg| sort -n | awk "NR%2==0") \${d}_b_tmp.gif;
            mv \${d}_b_tmp.gif \${d}_b.gif
        done


        sleep 10
    done'
    exec_on_window_no_log jupyter "cd ${HOME}/tmp/simulations"
    exec_on_window_no_log jupyter "jupyter-notebook"
fi
