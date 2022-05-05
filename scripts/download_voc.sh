#!/bin/bash

# Check that an argument was provided. Otherwise, print the usage of the program and terminate the script.
if (( $# != 1 )); then
  echo "Usage: ./download_voc.sh <dir_path>"
  echo -e "Download the PASCAL VOC 2007 (train + val + test) and 2012 (train + val) datasets.\n"
  echo "Arguments:"
  echo -e "\t<dir_path>: The path of the directory that will be created to store the PASCAL VOC data"
  exit 1
fi

# Create the directory where the PASCAL VOC dataset will be saved (if it doesn't exist).
mkdir -p $1

# Download the data, if they haven't already been downloaded (don't print a diagnostic message if the data have already
# been downloaded).
wget -nc -P $1 http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar \
               http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar \
               http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar > /dev/null 2>&1

# Extract the data from the .tar files (don't print the names of the extracted files)
tar -xvf $1/VOCtrainval_11-May-2012.tar -C $1 > /dev/null
tar -xvf $1/VOCtrainval_06-Nov-2007.tar -C $1 > /dev/null
tar -xvf $1/VOCtest_06-Nov-2007.tar -C $1 > /dev/null

# Execution successful
exit 0