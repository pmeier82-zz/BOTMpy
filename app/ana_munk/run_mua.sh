#!/bin/bash
#
# Script to start multi units analysis for passed exp, block, tetrode
#

# check for cl arguments
if [ $# -lt 3 ]
then
echo "Error - Pass at least three arguments or more triplets \"exp block tetrode\"!"
exit 1
fi

# start analysis sequentially
while [ "$1" != "" ]; do
do
  EXP=$i
  echo $EXP ${i}
  #nice -10 python ./munk_botm.py $arg 1>./botm_$arg.out 2>./botm_$arg.err
done
