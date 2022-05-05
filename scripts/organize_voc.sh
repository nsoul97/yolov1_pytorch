#!/bin/bash

# Check that an argument was provided. Otherwise, print the usage of the program and terminate the script.
if (( $# != 1 )); then
  echo "Usage: ./organize_voc.sh <dir_path>"
  echo  "Create a new directory 'VOC_Detection/' in the given directory where the images of the training and the test" \
        "set are separated in the following way:"

  echo
  echo  "VOC_Detection/"
  echo  "|-- train/"
  echo  "|   |-- images/"
  echo  "|   \`-- targets/"
  echo  "\`-- test/"
  echo  "|   |-- images/"
  echo  "|   |-- targets/"


  echo
  echo "The data of the VOC2007 and VOC2012 datasets are split in the following way in this new directory:
    - training set: train 2007 + val 2007 + train 2012 + val 2012
    - test set: test 2007"

  echo
  echo "The previous 'VOCdevkit/' directory is deleted."

  echo "Arguments:"
  echo -e "\t<dir_path>: The path of the directory that contains the PASCAL VOC data and, specifically, the" \
          "'VOCdevkit/' directory"
  exit 1
fi



# Create the folders as described above to separate the images and the annotations of the training set and the test set.
for dataset_part_dir in 'train' 'test'; do
    for xy_part_dir in 'images' 'targets'; do
        mkdir -p $(realpath -m $1/VOC_Detection/$dataset_part_dir/$xy_part_dir)
    done
done


# Move the images and the annotations to the 'PASCAL_VOC/train/' directory
for year in 2007 2012; do
    old_dir=$(realpath -m $1/VOCdevkit/VOC$year)
    new_dir_train=$(realpath -m $1/VOC_Detection/train)
    for img in $(cat $old_dir/ImageSets/Main/trainval.txt); do
        mv $old_dir/JPEGImages/$img.jpg $new_dir_train/images/
        mv $old_dir/Annotations/$img.xml $new_dir_train/targets/
    done
done


# Move the images and the annotations to the 'PASCAL_VOC/test/' directory
old_dir=$(realpath -m $1/VOCdevkit/VOC2007)
new_dir_test=$(realpath -m $1/VOC_Detection/test)
for img in $(cat $old_dir/ImageSets/Main/test.txt); do
        mv $old_dir/JPEGImages/$img.jpg $new_dir_test/images/
        mv $old_dir/Annotations/$img.xml $new_dir_test/targets/
done

# Delete the old VOCdevkit/ directory.
rm -rf $(realpath $1/VOCdevkit)

# Execution successful
exit 0