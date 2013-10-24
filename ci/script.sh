#!/bin/bash

echo "inside $0"

## TESTS

echo nosetests --exe -w /tmp -A "$NOSE_ARGS" pandas --show-skipped
nosetests --exe -w /tmp -A "$NOSE_ARGS" pandas --show-skipped

## EOF
true
