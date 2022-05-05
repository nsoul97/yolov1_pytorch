import os.path
import torch as th
import torchvision.transforms.functional as fT
from torchvision.utils import draw_bounding_boxes
from model import YOLOv1
from dataset import VOC_Detection
from evaluate import postprocessing
import PIL.Image as Image
from typing import Tuple
import matplotlib.pyplot as plt
import matplotlib

# Model Hyperparameters
S = 7
B = 2
D = 448

# Trained Model Path
TRAINED_MODEL_WEIGHTS = "/home/soul/Development/You Only Look Once - Unified, Real-Time Object " \
                        "Detection/checkpoints/trained_model_weights.pt"

# VOC Dataset Directory
PASCAL_VOC_DIR_PATH = "/media/soul/DATA/cv_datasets/PASCAL_VOC/VOC_Detection"

# Save Image Path
ASSETS_DIR = "/home/soul/Development/You Only Look Once - Unified, Real-Time Object Detection/assets"

# Compute Device (use a GPU if available)
DEVICE = 'cuda' if th.cuda.is_available() else 'cpu'

# Postprocessing Hyperparameters
PROB_THRESHOLD = 0.15
NMS_THESHOLD = 0.6
CONF_MODE = 'objectness'


global mv, save


def on_key_press(event: matplotlib.backend_bases.Event) -> None:
    """
    Set the mv global variable as +1 or -1 when the arrow keys are pressed. The mv flag is used to change the current
    image of the test set. When the Q is pressed, the program exits. When the key S is pressed, the global variable
    save is updated to True to save the current image. Otherwise, the global variables are updated for nothing to
    happen.

    :param event: An event that is triggered when a key is pressed
    """
    global mv, save
    if event.key == 'left':
        mv = -1
        save = False
    elif event.key == 'right':
        mv = +1
        save = False
    elif event.key == 's':
        mv = 0
        save = True
    elif event.key == 'q':
        exit(0)
    else:
        mv = 0
        save = False


def annotate_img(img: Image.Image,
                 bboxes: th.Tensor
                 ) -> Image:
    """
    Annotate the given image based on the given bounding boxes.
    The bounding box is plotted for each object of the image and the corresponding label is also written inside the box.

    :param img:  The PIL image to be annotated
    :param bboxes: The ground truth annotation data of the image
    :return: A PIL Image with the bounding boxes and their corresponding labels plotted.
    """

    img_tensor = fT.pil_to_tensor(img)
    bboxes_coords = bboxes[:, 2:]

    bboxes_class = bboxes[:, 0].long()
    objectness = bboxes[:, 1]
    text = [f'{VOC_Detection.index2label[bb_class_ind]}: {objectness[i] * 100:.1f}%' for i, bb_class_ind in
            enumerate(bboxes_class)]
    obj_clrs = [VOC_Detection.label_clrs[bb_class_ind] for bb_class_ind in bboxes_class]

    annotated_tensor = draw_bounding_boxes(img_tensor, bboxes_coords, text, width=4, font_size=20, colors=obj_clrs)
    annotated_img = fT.to_pil_image(annotated_tensor)
    return annotated_img


def update_plot(ax: matplotlib.axes.Axes,
                model: YOLOv1,
                img: Image.Image
                ) -> Image.Image:
    """
    Predict and plot the bounding boxes for the given image.

    :param ax: The axis where the annotated image will be plotted.
    :param model: The trained YOLOv1 (detection) model
    :param img: The PIL Image of the test set that will be fed to the YOLO model.
    :return: The input image annotated with the bounding box predictions
    """
    w, h = img.size
    x = fT.normalize(fT.to_tensor(fT.resize(img, (D, D))),
                     mean=[0.4549, 0.4341, 0.4010],
                     std=[0.2703, 0.2672, 0.2808]).unsqueeze(0).to(DEVICE)

    with th.no_grad():
        y = model(x)
    bboxes_pred = postprocessing(y,
                                 prob_threshold=PROB_THRESHOLD,
                                 conf_mode=CONF_MODE,
                                 nms_threshold=NMS_THESHOLD)

    # After postprocessing, the bounding box coordinates are scaled for a (D x D) image.
    bboxes_pred[:, [2, 4]] *= w / D
    bboxes_pred[:, [3, 5]] *= h / D

    img = annotate_img(img, bboxes_pred)
    ax.imshow(img)
    plt.show()

    return img


def setup_evaluation() -> Tuple[YOLOv1, VOC_Detection]:
    """
    Instantiate the model and the PASCAL VOC test dataset. The model's weights are loaded from the checkpoint file that
    was updated at the end of the training. The model will be used in the evaluation mode.

    :return: The trained YOLOv1 (detection) model and the PASCAL VOC test dataset.
    """
    model = YOLOv1(S=S,
                   B=B,
                   C=VOC_Detection.C).to(DEVICE)
    trained_model_weights = th.load(TRAINED_MODEL_WEIGHTS)
    model.load_state_dict(trained_model_weights)
    model.eval()

    test_dataset = VOC_Detection(root_dir=PASCAL_VOC_DIR_PATH,
                                 split='test')

    return model, test_dataset


def main():
    """
    Plot the bounding box prediction and the images interactively. To navigate in the test set, press the left/right
    arrow keys. To save the annotated image, press S. To terminate the program, press Q.
    """
    model, test_dataset = setup_evaluation()

    global mv, save
    plt.ion()
    fig, ax = plt.subplots()
    ax.axis('off')
    fig.canvas.mpl_disconnect(fig.canvas.manager.key_press_handler_id)
    fig.canvas.mpl_connect('key_press_event', on_key_press)

    i = 0
    img = test_dataset[i][0]
    annot_img = update_plot(ax, model, img)
    mv = 0
    save = False
    while True:
        if mv != 0:
            i = (i + mv) % len(test_dataset)
            img = test_dataset[i][0]
            annot_img = update_plot(ax, model, img)
        if save:
            path = os.path.join(ASSETS_DIR, f'annnot_img_{i}.jpg')
            annot_img.save(path)
        mv = 0
        save = False
        plt.waitforbuttonpress()


if __name__ == '__main__':
    main()
