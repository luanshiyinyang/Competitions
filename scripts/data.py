"""
Author: Zhou Chen
Date: 2020/1/8
Desc: desc
"""
from config import data_folder
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd


class DataSet(object):

    def __init__(self, root_folder):
        self.folder = root_folder
        self.df_desc = pd.read_csv(self.folder, encoding="utf8")

    def get_generator(self, da=True):
        if da:
            # 数据增强
            train_gen = ImageDataGenerator(rescale=1 / 255., validation_split=0.2, horizontal_flip=False, shear_range=0.2,
                                           width_shift_range=0.1)
        else:
            train_gen = ImageDataGenerator(rescale=1 / 255., validation_split=0.2, horizontal_flip=False,
                                           shear_range=0.2,
                                           width_shift_range=0.1)
        img_size = (224, 224)
        batch_size = 32
        train_generator = train_gen.flow_from_dataframe(dataframe=self.df_desc,
                                                        directory='.',
                                                        x_col='file_id',
                                                        y_col='label',
                                                        batch_size=batch_size,
                                                        class_mode='categorical',
                                                        target_size=img_size,
                                                        subset='training')
        valid_generator = train_gen.flow_from_dataframe(dataframe=self.df_desc,
                                                        directory=".",
                                                        x_col="file_id",
                                                        y_col="label",
                                                        batch_size=batch_size,
                                                        class_mode="categorical",
                                                        target_size=img_size,
                                                        subset='validation')
        return train_generator, valid_generator
