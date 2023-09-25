import pandas as pd
import tensorflow as tf
import emoji
import numpy as np
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.utils import pad_sequences
from keras.layers import Embedding, LSTM, Dense

# 读取数据集文件
data = pd.read_csv('/Users/CJJ/PycharmProjects/emo/data/emojify_data.csv')

# 将数据集拆分为句子和标签

#sentences = data['0'].values
sentences = data['French macaroon is so tasty'].values
labels = data['4'].values

# 将表情符号转换为数值标签
label_dict = {':heart:': 0,
              ':basketball:': 1,
              ':smile:': 2,
              ':disappointed:': 3,
              ':fork_and_knife:': 4}
#labels = np.array([label_dict[label] for label in labels])

# 将文本转换为数值序列
tokenizer = Tokenizer(num_words=10000, oov_token='<OOV>')
tokenizer.fit_on_texts(sentences)
sequences = tokenizer.texts_to_sequences(sentences)

# 填充所有序列为相同的长度
max_len = max([len(seq) for seq in sequences])
padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')

# 将标签转换为独热编码

"""
使用 to_categorical 函数将标签转换为独热编码，
其实就是将一个整数标签转换为一个向量形式的标签，其中只有标签对应的位置是 1，其他位置都是 0。这个过程可以通过以下步骤实现：

首先，统计标签中共有多少个不同的取值，这个数量就是独热编码的维度。
在本例中，我们有五种不同的标签，因此独热编码的维度为 5。

然后，将每个整数标签映射到一个唯一的索引位置。
在本例中，我们使用一个字典 label_dict 将每个表情符号映射到一个唯一的整数编号，这个编号就是标签的整数表示。

接下来，使用 Keras 中的 to_categorical 函数将整数标签转换为独热编码。
这个函数接收两个参数：labels 表示待转换的整数标签，num_classes 表示独热编码的维度。

在本例中，我们将 labels 参数设置为 labels 变量，num_classes 参数设置为 5，即独热编码的维度。
最后，to_categorical 函数会将整数标签转换为独热编码，并返回一个二维的 Numpy 数组。

这个数组的每一行都是一个独热编码，其中只有一个位置是 1，其他位置都是 0。
这个位置的索引就是原始整数标签在字典中对应的编号。可以将这个数组作为深度学习模型的标签，用于训练和预测。
"""
one_hot_labels = to_categorical(labels)

# 构建模型
model = keras.Sequential()
model.add(Embedding(10000, 8))
model.add(LSTM(10))
model.add(Dense(5, activation='softmax'))

# 编译模型
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# 训练模型
model.fit(padded_sequences, one_hot_labels, epochs=50, verbose=1)

# 对新句子进行预测
test_sentences = [
    'I\'m so happy today',
    'You\'re the best!',
    'Have a wonderful day',
    'Let\'s have lunch',
    'I bought a basketball yesterday'
]
test_sequences = tokenizer.texts_to_sequences(test_sentences)
padded_test_sequences = pad_sequences(test_sequences, maxlen=max_len, padding='post', truncating='post')
predictions = model.predict(padded_test_sequences)



predicted_emoji_indices = predictions.argmax(axis=1)
predicted_emojis = [list(label_dict.keys())[list(label_dict.values()).index(idx)] for idx in predicted_emoji_indices]
predicted_emoji_symbols = [emoji.emojize(predicted_emoji) for predicted_emoji in predicted_emojis]
print(predicted_emoji_symbols)