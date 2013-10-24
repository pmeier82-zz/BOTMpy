#!/bin/bash

echo "inside $0"

## DIAGNOSTICS

echo "before_install"
echo $VIRTUAL_ENV
df -h
date
python -V

## PACKAGE INDEX

sudo apt-get update -qq

## EOF
true
