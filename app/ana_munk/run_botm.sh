#!/bin/bash
#
# Script to start botm analysis for passed analysis ids
#

# check for cl arguments
if [ $# -eq 0 ]
then
echo "Error - Pass at least one analysis id!"
exit 1
fi

# start analysis sequentially
for arg in $*
do  
  nice -10 python ./munk_botm.py $arg 1>./log/botm_$arg.out 2>./log/botm_$arg.err
done