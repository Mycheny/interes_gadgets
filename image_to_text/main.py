# -*- coding: utf-8 -*- 
# @Time 2020/6/23 11:43
# @Author wcy
import cv2
import numpy as np
from sklearn.cluster import KMeans


def recreate_image(codebook, labels, mask, w, h):
    """从代码簿和标签中重新创建（压缩）图像"""
    mask_flatten = mask.flatten()
    labels_ = np.zeros_like(mask_flatten, dtype=np.int) - 1
    labels_[mask_flatten.astype(np.bool)] = labels
    labels_ = np.reshape(labels_, mask.shape)
    image = np.zeros((h, w))
    image[labels_ == -1] = 0
    for i, rgb in enumerate(range(codebook.shape[0])):
        image[labels_ == i] = rgb
    return image.astype(np.uint8)


def narray_to_text(image, w=28, h=None, mask=None):
    # pixels = ["██", "◆", "▲", "☀", "✔", "∵", "▒▒", "▓▓", "※", "…", "", ]
    # pixels = ["██", "▇▇", "▆▆", "▅", "▄▄", "▃▃", "▂▂", "▁▁", "  "]
    # pixels = ["███", "☀☀☀", "**", "..."]
    # pixels = ["口 ", "爱 ", "你 ", "一 "]
    # pixels = ["我 ", "爱 ", "你 "]
    pixels = ["I   ", "love", "you ", "!   "]
    if len(image.shape)==3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_h, img_w = image.shape
    if h is None:
        h = int((w*img_h)/img_w+0.5)
    image = cv2.resize(image, (w, h))
    if mask is None:
        mask = np.ones(image.shape)
    image_array = image.flatten()
    image_array = np.expand_dims(image_array, axis=-1)
    kmeans = KMeans(n_clusters=(len(pixels)), random_state=0).fit(image_array)
    labels = kmeans.predict(image_array)
    image_ = recreate_image(kmeans.cluster_centers_, labels, mask, w, h)
    frame = np.zeros_like(image, np.str)
    frame = frame.tolist()
    for i, pixel in enumerate(pixels):
        y, x = np.where(image_ == i)
        for y, x in zip(y, x):
            frame[y][x]= pixel
    return np.array(frame)


if __name__ == '__main__':
    image = cv2.imread("./image/xiaohui_h.jpg", 0)
    texts = narray_to_text(image)
    with open("./result.txt", "w", encoding="utf-8") as f:
        for text in texts:
            f.write(f"{''.join(text)}\n")
