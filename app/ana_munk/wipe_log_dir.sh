#!/bin/bash
#
# wipes the contents the log dir, which is assumed to be located at ./log
#

# try wiping the contents of ./log
if [ -d ./log ]
then
  echo "found directory"
  rm -rf ./log/*
  echo "wiped contents"
fi