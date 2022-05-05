import os
import argparse
import xml.etree.ElementTree as Et


def parse_args():
    parser = argparse.ArgumentParser(description="Simplify the .xml format of the targets to preserve only the label "
                                                 "and the coordinates of the bounding boxes for each image. For each "
                                                 "bounding box in an image a new .csv file with the same name is "
                                                 "created. The first row of the .csv file contains the header and the "
                                                 "next rows have the format '<object>,<xmin>,<ymin>,<xmax>,<ymax>'."
                                                 "The old .xml annotation files are deleted.")

    parser.add_argument('dataset_path', type=str, help='The base path of the PASCAL VOC dataset (where the .tar files '
                                                       'were downloaded).')

    args = parser.parse_args()
    return args.dataset_path


def simplify_targets(dataset_path: str) -> None:
    """
    Simplify the .xml format of the target to preserve only the label and the coordinates of the bounding boxes in the
    .csv target file of an image. The .xml files are removed. The difficult objects are ignored both for the training
    and the test set.

    :param dataset_path: The base path of the PASCAL VOC dataset (where the .tar files were downloaded).
    """
    for dataset_part_dir in ['train', 'test']:
        annot_dir_name = os.path.join(dataset_path, "VOC_Detection", dataset_part_dir, "targets")
        for annot_file_name in os.listdir(annot_dir_name):

            csv_filename = os.path.join(annot_dir_name, f'{annot_file_name[:-4]}.csv')
            xml_filename = os.path.join(annot_dir_name, annot_file_name)

            with open(csv_filename, 'w') as csv_file:
                csv_file.write("object,xmin,ymin,xmax,ymax")
                with open(xml_filename, 'r') as xml_file:
                    for obj in Et.parse(xml_file).getroot().findall('object'):
                        difficult_flag = obj.find('difficult').text
                        if difficult_flag == '0':
                            label = obj.find('name').text
                            bbox_dict = {bndbox.tag: bndbox.text for bndbox in obj.find('bndbox')}
                            csv_file.write(f"\n{label},"
                                           f"{bbox_dict['xmin']},{bbox_dict['ymin']},"
                                           f"{bbox_dict['xmax']},{bbox_dict['ymax']}")
            os.remove(xml_filename)


if __name__ == '__main__':
    dataset_path = parse_args()
    simplify_targets(dataset_path)
