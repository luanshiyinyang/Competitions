"""
Author: Zhou Chen
Date: 2020/1/8
Desc: desc
"""
from data import DataSet
from model import CNN, ResNet_pretrained
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
from utils import plot_history
import os

if not os.path.exists("../models/"):
    os.mkdir("../models/")


# 数据
ds = DataSet('../data/')
train_generator, valid_generator = ds.get_generator(batch_size=64)
img_shape = (64, 64, 1)
# 模型
model_cnn = CNN(input_shape=img_shape)
model_resnet = ResNet_pretrained(input_shape=img_shape)
# 训练
optimizer_cnn = Adam(lr=3e-4)
optimizer_resnet = Adam(lr=3e-4)
callbacks_cnn = [
    ModelCheckpoint('../models/cnn_best_weights.h5', monitor='val_loss', save_best_only=True, verbose=1, save_weights_only=True),
    # EarlyStopping(monitor='val_loss', patience=5)
]

callbacks_resnet = [
    ModelCheckpoint('../models/resnet_best_weights.h5', monitor='val_loss', save_best_only=True, verbose=1, save_weights_only=True),
    # EarlyStopping(monitor='val_loss', patience=5)
]

model_cnn.compile(optimizer=optimizer_cnn, loss='categorical_crossentropy', metrics=['accuracy'])
model_resnet.compile(optimizer=optimizer_resnet, loss='categorical_crossentropy', metrics=['accuracy'])

epochs = 20
history_cnn = model_cnn.fit_generator(train_generator,
                              steps_per_epoch=train_generator.n//train_generator.batch_size,
                              validation_data=valid_generator,
                              validation_steps=valid_generator.n//valid_generator.batch_size,
                              epochs=epochs,
                              callbacks=callbacks_cnn
                             )
history_resnet = model_cnn.fit_generator(train_generator,
                              steps_per_epoch=train_generator.n//train_generator.batch_size,
                              validation_data=valid_generator,
                              validation_steps=valid_generator.n//valid_generator.batch_size,
                              epochs=epochs,
                              callbacks=callbacks_resnet
                             )

plot_history([history_cnn, history_resnet])


