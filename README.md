# 文本分类


## 简介
这是达观在2018年举办的一个文本分类比赛，是一场经典的NLP比赛，关于NLP赛的思路在[之前的博客](https://zhouchen.blog.csdn.net/article/details/103863618)中提到过，目前这场比赛已经结束，但是仍旧可以在DC上[提交成绩](https://www.dcjingsai.com/common/cmpt/%E2%80%9C%E8%BE%BE%E8%A7%82%E6%9D%AF%E2%80%9D%E6%96%87%E6%9C%AC%E6%99%BA%E8%83%BD%E5%A4%84%E7%90%86%E6%8C%91%E6%88%98%E8%B5%9B_%E7%AB%9E%E8%B5%9B%E4%BF%A1%E6%81%AF.html)，作为一个demo的比赛了。本文将简要对该比赛的思路进行介绍，采用传统方法和深度方法提交baseline模型。


## 数据探索
数据集可以直接到[官网](https://www.dcjingsai.com/common/cmpt/%E2%80%9C%E8%BE%BE%E8%A7%82%E6%9D%AF%E2%80%9D%E6%96%87%E6%9C%AC%E6%99%BA%E8%83%BD%E5%A4%84%E7%90%86%E6%8C%91%E6%88%98%E8%B5%9B_%E7%AB%9E%E8%B5%9B%E4%BF%A1%E6%81%AF.html)下载，下载后解压文件可以得到训练集和测试集，均为CSV格式的表格文件，可以采用Pandas进行读取和分析。

![](./assets/dataset.png)

对训练集的数据进行初步探索，结果如下。其中id为文本标识码，article为文本字表示（每个数字对应一个汉字，不知道字表无法还原，这是为了保护文本中的隐私信息，称为脱敏操作，该操作不允许建模预测等），word_seg是文本的词表示，每个数字编号了一个单词，class为该文本的类别标签号。

![](./assets/ds_head.png)

为了后面模型的设计，有必要知道文本的长度范围是什么（下图探索词长度，即一个文本多少个词）。文本平均含有716个单词，是一个典型的长文本分类。

![](./assets/word_length.png)

对词信息进行计数统计，结果如下。高频词出现了500万次，总共有875129个词。当然，低频词也有很多，这里不多说明，具体见文末Github地址。

![](./assets/word_count.png)

对字的处理类似上面对于词的处理，不多说明，不过在NLP中词的信息远远多于字的信息量。

最后，可以看看**标签的分布**，可以使用Pandas接口轻易完成。

![](./assets/label.png)


经过数据探索，不难发现这是一个长文本分类问题，且样本类别分布不均衡，同时词量特别大，会导致特征数量很多。


## Pipeline制定（传统方法baseline）
在这一部分会进行数据集的特征工程，构建模型，预测提交结果。

### 特征工程
首先采用传统方法提取文本特征，分别采用TFIDF和N-Gram，这两种方法都是在NLP中比较传统的基于统计的文本特征提取思路，其原理这里不多赘述，这里使用sklearn中封装好的API，其具体使用可以参考scikit-learn的官方文档。

使用下面的代码生成每个文本的特征向量，生成的特征向量是稀疏矩阵，scikit-learn中模型支持稀疏矩阵的输入。 

```python
from sklearn.feature_extraction.text import TfidfVectorizer

word_vec = TfidfVectorizer(analyzer='word',
            ngram_range=(1,2),
            min_df=3,  # 低频词
            max_df=0.9,  # 高频词
            use_idf=True,
            smooth_idf=True, 
            sublinear_tf=True)

train_doc = word_vec.fit_transform(df_train['word_seg'])
test_doc = word_vec.transform(df_test['word_seg'])
```

### 模型构建
这一部分先是采用最基本的机器学习分类模型---逻辑回归进行模型的训练及测试集的预测。
```python
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(C=4) 
clf.fit(train_doc, df_train['label'])
test_prob = clf.predict_proba(test_doc)
test_pred = np.argmax(test_prob, axis=1)
df_test['class'] = lb.inverse_transform(test_pred)
df_test[["id","class"]].to_csv("submission.csv", index=False, header=True, encoding='utf-8')
```

这个baseline的提交成绩如下图，采用的metric是F1得分，这个成绩的排名是648位。

![](./assets/submission1.png)

### 优化思路
后续的优化都是基于上面的baseline进行的，本部分具体代码见文末Github。

**首先**，上述的特征工程均只使用了词的信息，没有使用字的信息，可以使用字的特征组合词的特征从而达到充分利用数据的目的，这里简单的将两种特征向量横向堆叠，这样会出现特征维度太高的问题，需要进行降维。

**接着**，逻辑回归毕竟只是一个基本的分类模型，其实可以使用更强的集成模型如LightGBM、XGBoost等，这里使用LightGBM进行建模（关于LGB调参技巧本文不多说明）。

上述的思路其实有一个问题，我们始终没有在线下得到模型的效果（**事实上，正规的比赛对提交次数都是有限制的，不可能优化一次代码就提交一次观察线上得分变化，这会造成部分人的刷分，必须在线下进行模型评估。**），通常，我们采用构建验证集的方法在线下进行模型评估（k折交叉验证得到的验证得分更加合适，但是资源消耗大）。

**这一部分主要是使用集成模型且进行交叉验证，得到平均的预测结果。当然，这部分需要大量人工的特征工程和模型调参，后面会介绍效果更佳显著的深度学习方法。**


## Pipeline制定（深度方法）
在这一部分不会过度强调特征工程、模型等步骤，因为在深度学习方法中主要目的是构建端到端的一个应用系统（如文本类别识别系统）。

### 数据准备
首先，将文本转化为序列（分词），这个过程会建立词表，这样每个文本变为了一个序列。（模型期待输入是固定的维度所以对不等长文本需要进行截断和补全，截取或者补全后的序列长度视情况而定）

同时，需要对标签进行onehot编码以便于使用softmax进行输出层激活且计算loss。

### 文本特征
词有很多表示方法，最简单的是onehot编码单词，每个单词就是其对应的onehot编码，但是onehot有一个显著的特征---无法衡量不同单词之间的距离（因为距离都是相同的）且维度很高。后来提出的Word2Vec方法可以构造低维有距的词向量，它的产生有两种策略分别为CBOW和Skip-gram，具体的理论这里不多赘述。

这里将输入的文本词序列输入模型构建词嵌入（word embedding），利用词表和词向量构建词嵌入可以大幅减少内存消耗。

### 模型构建
下面构建整个深度模型，使用双向GRU+池化层构建网络，全连接层作为分类器。
```python
def build_model(sequence_length, embedding_weight, class_num):
    content = Input(shape=(sequence_length, ), dtype='int32')
    embedding = Embedding(
        name='word_embedding',
        input_dim=embedding_weight.shape[0],
        weights=[embedding_weight],
        output_dim=embedding_weight.shape[1],
        trainable=False
    )
    x = SpatialDropout1D(0.2)(embedding(content))
    x = Bidirectional(GRU(200, return_sequences=True))(x)
    x = Bidirectional(GRU(200, return_sequences=True))(x)
    
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    conc = concatenate([avg_pool, max_pool])
    x = Dense(1000)(conc)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(500)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    output = Dense(19, activation='softmax')(x)
    model = tf.keras.models.Model(inputs=content, outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
```

### 模型训练
训练采用交叉验证，并综合多次的预测结果以获得更好的模型表现。（综合的方法是10折的预测取均值，即线性加权。）
```python
kf = KFold(n_splits=10, shuffle=True, random_state=2019)
train_pre_matrix = np.zeros((df_train.shape[0], 19))
test_pre_matrix = np.zeros((10, df_test.shape[0], 19))
cv_scores = []

for i, (train_index, valid_index) in enumerate(kf.split(train_)):
    x_train, x_valid = train_[train_index, :], train_[valid_index, :]
    y_train, y_valid = train_label[train_index], train_label[valid_index]
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(64)
    valid_ds = tf.data.Dataset.from_tensor_slices((x_valid, y_valid)).batch(64)
    test_ds = tf.data.Dataset.from_tensor_slices((test_, np.zeros((test_.shape[0], 19)))).batch(64)
    
    model = build_model(1800, embedding_matrix, 19)
    model.fit(train_ds, epochs=30, validation_data=valid_ds, verbose=1)
    
    valid_prob = model.predict(valid_ds)
    valid_pred = np.argmax(valid_prob, axis=1)
    valid_pred = lb.inverse_transform(valid_pred)
    y_valid = np.argmax(y_valid, axis=1)
    y_valid = lb.inverse_transform(y_valid)
    f1_score = f1_score(y_valid, valid_pred, average='macro')
    print("F1 score", f1_score)
    train_pre_matrix[valid_index, :] = valid_prob
    test_pre_matrix[i, :, :] = model.predict(test_ds)
    del model
    gc.collect()
    tf.keras.backend.clear_session()

np.save('test.npy', test_pre_matrix)
```

### 结果提交
将综合后的预测结果提交到比赛平台，可以看到，得分如下，排名从传统方法的600多到达了前100，说明深度方法的学习能力是很强的。


## 补充说明
本文简要以达观的文本分类为例，讲述了NLP赛的如今主流思路，2019年达观举办了另外一场比赛，有兴趣也可以参与。本文所有代码开源于[我的Github仓库](https://github.com/luanshiyinyang/Competitions/tree/TextClassification)，欢迎star或者fork。

