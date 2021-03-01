#!/bin/bash

date
echo before
free

sync
echo 3 > /proc/sys/vm/drop_caches

echo after
free
