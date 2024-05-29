import cv2
import numpy as np

def tensor_to_cv2image(tensor, remove_padding=True):
    image = tensor.numpy()
    image = image.transpose((1, 2, 0))
    image = image * 255
    if remove_padding:
        image_ = np.sum(image, -1)
        image_h = np.sum(image_, 1)
        if 0 in image_h:
            h_border = np.min(np.where(image_h == 0)[0])
        else:
            h_border = image.shape[0]
        image_w = np.sum(image_, 0)
        if 0 in image_w:
            w_border = np.min(np.where(image_w == 0)[0])
        else:
            w_border = image.shape[1]
        image = image[:h_border, :w_border]
    image = image.astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image