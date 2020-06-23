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
    d = codebook.shape[1]
    image = np.zeros((h, w, d))
    image[labels_ == -1, :] = [0, 255, 0]
    for i, rgb in enumerate(codebook):
        image[labels_ == i] = rgb
    return image.astype(np.uint8)


def recreate_image_gray(codebook, labels, mask, w, h):
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


def narray_to_text(image, w=128, h=None, mask=None):
    pixels = ["我 ", "爱 ", "你 "]
    # pixels = ["I   ", "love", "you ", "!   "]
    if len(image.shape)==3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_h, img_w = image.shape
    if h is None:
        h = int((w*img_h)/img_w+0.5)
    image = cv2.resize(image, (w, h))
    if mask is None:
        mask = np.ones(image.shape)
    # image = cv2.blur(image, (2, 2))
    # image = cv2.GaussianBlur(image, (3, 3), 0)
    # image = cv2.bilateralFilter(image, 40, 75, 75)
    image = cv2.medianBlur(image, 5)
    image_array = image.flatten()
    image_array = np.expand_dims(image_array, axis=-1)
    kmeans = KMeans(n_clusters=(len(pixels)), random_state=0).fit(image_array)
    labels = kmeans.predict(image_array)
    image_ = recreate_image_gray(kmeans.cluster_centers_, labels, mask, w, h)
    cv2.imshow("", (image_-image_.min())/(image_.max()-image_.min()))
    cv2.waitKey(0)
    frame = np.zeros_like(image, np.str)
    frame = frame.tolist()
    for i, pixel in enumerate(pixels):
        y, x = np.where(image_ == i)
        for y, x in zip(y, x):
            frame[y][x]= pixel
    return np.array(frame)


def deal_image(image, n_clusters=4, w=128):
    img_h, img_w, _ = image.shape
    h = int((w * img_h) / img_w + 0.5)
    image = cv2.resize(image, (w, h))
    mask = np.ones(image.shape[:2])
    h, w, _ = image.shape
    image_array = image[mask.astype(np.bool)]
    kmeans = KMeans(n_clusters=(n_clusters), random_state=0).fit(image_array)
    labels = kmeans.predict(image_array)
    image = recreate_image(kmeans.cluster_centers_, labels, mask, w, h)
    cv2.imshow("", image)
    cv2.waitKey(0)


if __name__ == '__main__':
    image = cv2.imread("./image/xiaohui.jpg")
    deal_image(image)

    texts = narray_to_text(image)
    with open("./result.txt", "w", encoding="utf-8") as f:
        for text in texts:
            f.write(f"{''.join(text)}\n")
