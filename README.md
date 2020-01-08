# 人脸表情识别


## 简介
这是科赛网曾今的一次计算机视觉类的分类赛，属于一个视觉基础任务。关于人脸表情识别，Kaggle等平台也举办过相关的比赛，难度并不算大，但是对技巧的要求比较高。本文详述该比赛的主要步骤，并构建多个模型对比效果。


## 数据探索
这部分主要是对数据集的简单探索，由于计算机视觉类比赛数据多为图片或者视频格式，无需像挖掘赛那样进行比较复杂的EDA。这部分的代码可以在文末给出的Github仓库的`EDA.ipynb`文件中找到。

数据集的目录形式如下，由于本比赛只需要将五种区分度比较明显的表情识别出来，所以共五个文件夹，每个文件夹内的图片为一类。

![](./assets/dataset.png)

首先，生成了如下的csv格式的数据集说明文件如下，包含两列，分别为文件的相对路径和标签。对该说明文件进行分析更加的方便，因为这个文件包含大多数数据集的信息，且Pandas提供了很多分析的API，很容易对表格数据处理。

![](./assets/data_desc.png)

首先观察类别的分布，很遗憾，数据的类别分布如下。这是因为有的表情数据确实很少，尽管我们确实希望数据的分布是均衡的，但是这里的影响也不是很大，后面通过数据增强，会对这个情况有所改善。

![](./assets/distribution.png)

随后，随机采样10张图片展示并显示标签，结果如下。通过这一步的观察，不难发现，其实图片中不仅仅是目标-人脸表情，有很多背景干扰因素，这就要求需要进行人脸检测及图片裁减了，也就是输入深度模型中的不应该是原图而应该是裁减后的人脸图片。

![](./assets/data_sample.png)


## 数据预处理
这部分主要对图片进行人脸截取，实现的原理是基于目标检测的思路，分为传统方法和深度方法，为了更快得到结果，这里使用软件包`face_recognition`进行人脸检测（**注意处理单图片多人脸情况**），这是一个非常简单的封装好的人脸识别库，可以通过pip安装（若安装出错一般是dlib问题，安装dlib的我回来文件即可）。

其人脸检测的结果保存到本地后结果如下。

![](./assets/data_generated.png)

核心代码如下，具体代码见文末Github。
```python
def preprocess():
    categories = os.listdir(data_folder)
    for category in categories:
        in_path = os.path.join(data_folder, category)
        out_path = os.path.join(generate_folder, category + '_face')
        if not os.path.exists(out_path):
            os.mkdir(out_path)
        for file in glob(in_path + '/*.jpg'):
            file_name = file.split('\\')[-1]
            print(file_name)
            img = face_recognition.load_image_file(file)
            if max(img.shape) > 2000:
                if img.shape[0] > img.shape[1]:
                    img = cv2.resize(img, (2000, int(2000 * img.shape[1] / img.shape[0])))
                else:
                    img = cv2.resize(img, (int(2000 * img.shape[0] / img.shape[1]), 2000))
            locations = face_recognition.face_locations(img)  # 人脸检测，大部分为单个，但也有多个检测结果
            if len(locations) <= 0:
                print("no face")
            else:
                for i, (a, b, c, d) in enumerate(locations):
                    image_split = img[a:c, d:b, :]
                    image_split = scale_img(image_split)
                    Image.fromarray(image_split).save(os.path.join(out_path, file_name + '_{}.png'.format(i)))
```


## 数据加载
由于数据集已经被处理为很规范的数据集格式，这里直接调用TF2中Keras接口解析数据集，返回一个迭代器，同时在该接口中设置一些数据增强手段，这里主要使用水平翻转。


## 模型构建


## 补充说明
真正构建实际系统是可以在模型应用的预测端进行一个增广预测，将得到的结果进行加权从而得到真正的预测结果，具体可以参考我之前的[项目](https://github.com/luanshiyinyang/ExpressionRecognition)。

