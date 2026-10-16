#!/bin/bash
# usage: launch.sh <logname> <command...>   (runs detached on the pod; log in logs/<logname>.log)
cd /workspace/weak-ablation && source env.sh
name=$1; shift
setsid nohup bash -c "$*" > logs/$name.log 2>&1 < /dev/null &
echo "launched $name pid $!"
