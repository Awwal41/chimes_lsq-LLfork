#!/bin/bash

# Removes files/folders created during a CMake build
# Run with:
# ./uninstall.sh
# or 
# ./uninstall.sh <install prefix (full path)>

PREFX=${1-""} # Empty by default


for i in `cat build/install_manifest.txt 2>/dev/null`
do
	echo "Removing installed file $i"
	rm -f $i
done

echo "Removing directory build"

rm -rf build

if [ ! -z $PREFX ] ; then 
	echo "Removing files installed at $PREFX"
	echo "This runs the command \"rm -rf $PREFX\""
	echo "Are you sure you want to proceed?"
	read -p "Press enter to continue"
	
	rm -rf $PREFX
fi
