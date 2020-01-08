"""
Author: Zhou Chen
Date: 2020/1/8
Desc: desc
"""
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd


class DataSet(object):

    def __init__(self, root_folder):
        self.folder = root_folder
        self.df_desc = pd.read_csv(self.folder + 'description.csv', encoding="utf8")

    def get_generator(self, batch_size=32, da=True):
        if da:
            # 数据增强
            train_gen = ImageDataGenerator(rescale=1 / 255., validation_split=0.25, horizontal_flip=True)
        else:
            train_gen = ImageDataGenerator(rescale=1 / 255., validation_split=0.25, horizontal_flip=False)
        img_size = (64, 64)
        train_generator = train_gen.flow_from_dataframe(dataframe=self.df_desc,
                                                        directory='.',
                                                        x_col='file_id',
                                                        y_col='label',
                                                        batch_size=batch_size,
                                                        class_mode='categorical',
                                                        target_size=img_size,
                                                        color_mode='grayscale',
                                                        subset='training')
        valid_generator = train_gen.flow_from_dataframe(dataframe=self.df_desc,
                                                        directory=".",
                                                        x_col="file_id",
                                                        y_col="label",
                                                        batch_size=batch_size,
                                                        class_mode="categorical",
                                                        target_size=img_size,
                                                        color_mode='grayscale',
                                                        subset='validation')
        return train_generator, valid_generator
