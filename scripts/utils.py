"""
Author: Zhou Chen
Date: 2020/1/8
Desc: desc
"""
import matplotlib.pyplot as plt
import numpy as np
plt.style.use('fivethirtyeight')


def plot_history(his):
    cnn_his = his[0].history
    resnet_his = his[1].history
    plt.figure(figsize=(18, 6))
    plt.subplot(1, 2, 1)
    plt.plot(np.arange(len(cnn_his['accuracy'])), cnn_his['accuracy'], label="training accuracy")
    plt.plot(np.arange(len(cnn_his['val_accuracy'])), cnn_his['val_accuracy'], label="validation accuracy")
    plt.title("CNN")
    plt.legend(loc=0)

    plt.subplot(1, 2, 2)
    plt.plot(np.arange(len(resnet_his['accuracy'])), resnet_his['accuracy'], label="training accuracy")
    plt.plot(np.arange(len(resnet_his['val_accuracy'])), resnet_his['val_accuracy'], label="validation accuracy")
    plt.title("ResNet50")
    plt.legend(loc=0)

    plt.savefig("his.png")
    plt.show()
