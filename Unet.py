from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import os
from PIL import Image
import numpy as np
import pandas as pd


lst_img = os.listdir(r'C:\Users\MASSON\Desktop\Palm\img')
imgs = [r'C:\Users\MASSON\Desktop\Palm\img'+"\\"+x for x in lst_img]
imgs.sort()
lst_masks = os.listdir(r'C:\Users\MASSON\Desktop\Palm\masks')
masks = [r'C:\Users\MASSON\Desktop\Palm\masks'+"\\"+x for x in lst_masks]
masks.sort()

y = np.zeros((57, 128, 128,1), dtype=np.float32)
X = np.zeros((57,128, 128,3), dtype=np.float32)


for i in range(len(masks)):
    xi = Image.open(imgs[i])
    xi = np.array(xi.resize((128,128)))
    X[i] = xi
    yi = Image.open(masks[i])
    yi = np.array(yi.resize((128,128)))
    y[i] = np.expand_dims(yi,axis=-1)
#
#
X = X/255
#
# def down_block(x, filters, kernel_size=(3, 3), padding="same", strides=1):
#     c = tf.keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(x)
#     c = tf.keras.layers.BatchNormalization()(c)
#     c = tf.keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
#     c = tf.keras.layers.BatchNormalization()(c)
#     p = tf.keras.layers.MaxPool2D((2, 2), (2, 2))(c)
#     return c, p
#
# def up_block(x, skip, filters, kernel_size=(3, 3), padding="same", strides=1):
#     us = tf.keras.layers.UpSampling2D((2, 2))(x)
#     concat = tf.keras.layers.Concatenate()([us, skip])
#     c = tf.keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(concat)
#     c = tf.keras.layers.BatchNormalization()(c)
#     c = tf.keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
#     c = tf.keras.layers.BatchNormalization()(c)
#
#     return c
#
# def bottleneck(x, filters, kernel_size=(3, 3), padding="same", strides=1):
#     c = tf.keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(x)
#     c = tf.keras.layers.BatchNormalization()(c)
#     c = tf.keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
#     c = tf.keras.layers.BatchNormalization()(c)
#
#     return c
#
# def UNet():
#     f = [16, 32, 64, 128, 256]
#     inputs = tf.keras.layers.Input((480, 720, 3))
#
#     p0 = inputs
#     c1, p1 = down_block(p0, f[0]) #128 -> 64
#     c2, p2 = down_block(p1, f[1]) #64 -> 32
#     c3, p3 = down_block(p2, f[2]) #32 -> 16
#     c4, p4 = down_block(p3, f[3]) #16->8
#
#     bn = bottleneck(p4, f[4])
#
#     u1 = up_block(bn, c4, f[3]) #8 -> 16
#     u2 = up_block(u1, c3, f[2]) #16 -> 32
#     u3 = up_block(u2, c2, f[1]) #32 -> 64
#     u4 = up_block(u3, c1, f[0]) #64 -> 128
#
#     outputs = tf.keras.layers.Conv2D(1, (1, 1), padding="same", activation="sigmoid")(u4)
#     model = tf.keras.models.Model(inputs, outputs)
#     return model
#
# model = UNet()
# model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["acc",tf.keras.metrics.MeanIoU(2)])
#
# history = model.fit(x=X,y=y,epochs = 5,batch_size = 1)
#
# y_pred = model.predict(np.expand_dims(X[0],axis=0))
#

model_new = tf.keras.models.load_model(r'C:\Users\MASSON\Downloads\model',custom_objects = {"dice_loss_plus_1binary_focal_loss":sm.losses.DiceLoss()+1*sm.losses.BinaryFocalLoss(),"iou_score":sm.metrics.IOUScore(threshold=0.5),"f1-score":sm.metrics.FScore(threshold=0.5)})


model_new2 = tf.keras.models.load_model(r'C:\Users\MASSON\Downloads\best_model3',custom_objects = {"dice_loss_plus_1binary_focal_loss":sm.losses.DiceLoss()+1*sm.losses.BinaryFocalLoss(),"iou_score":sm.metrics.IOUScore(threshold=0.5),"f1-score":sm.metrics.FScore(threshold=0.5)})


import cv2


while True:
for i in range(len(X)):

    y_pred = model_new.predict(np.expand_dims(X[i],axis=0))
    y_pred[y_pred<0.5] = 0
    y_pred[y_pred>0.5] = 1
    bWx = cv2.cvtColor(X[i].squeeze(),cv2.COLOR_RGB2GRAY)
    addd = cv2.hconcat([bWx,bWx])
    plt.imshow(addd,cmap='gray')
    add2 = cv2.hconcat([y_pred.squeeze(),y[i].squeeze()])
    plt.imshow(add2,cmap='jet',alpha=0.2,interpolation='None')
    plt.show()
    plt.pause(0.001)
    plt.clf()

x_pred = Image.open(r'C:\Users\MASSON\Desktop\Palm\img\0.png')
# x_pred = x_pred.resize((320,320))
y_test = model_new2.predict(np.expand_dims(x_pred,axis=0))
plt.figure()
plt.imshow(x_pred)
plt.figure()
plt.imshow(y_test.squeeze())
plt.show()

