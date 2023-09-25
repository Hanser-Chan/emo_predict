import tensorflow as tf
import emoji
import pandas as pd
import numpy as np
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras import layers

data = pd.read_csv("/Users/CJJ/PycharmProjects/emo/data/emojify_data.csv")



# 训练数据

# sentences = [
#     'I love you',
#     'Let\'s go for a walk',
#     'Have a nice day',
#     'Congratulations on the promotion',
#     'I\'m sorry to hear that'
# ]
#emojis = [':heart:', ':basketball:', ':smile:', ':disappointed:', ':fork_and_knife:']

label_dict = {':heart:': 0,
              ':basketball:': 1,
              ':smile:': 2,
              ':disappointed:': 3,
              ':fork_and_knife:': 4}

# 将文本转换为数值序列
tokenizer = Tokenizer(num_words=10000, oov_token='<OOV>')
tokenizer.fit_on_texts(sentences)
sequences = tokenizer.texts_to_sequences(sentences)

# Create a tokenizer for the emojis
emoji_tokenizer = Tokenizer()
emoji_tokenizer.fit_on_texts(emojis)
emoji_sequences = emoji_tokenizer.texts_to_sequences(emojis)
emoji_labels = [seq[0] - 1 for seq in emoji_sequences]  # Subtract 1 to have labels starting from 0

# 填充所有序列为相同的长度
max_len = max([len(seq) for seq in sequences])
padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')

# 构建模型
model = keras.Sequential()
model.add(layers.Embedding(10000, 8))
model.add(layers.LSTM(10))
model.add(layers.Dense(5, activation='softmax'))

# 编译模型
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# 训练模型
x = np.array(padded_sequences)
y = np.array(emoji_labels)
model.fit(x, y, epochs=50, verbose=1)

# 对新句子进行预测
test_sentences = [
    'I\'m so happy today',
    'You\'re the best!',
    'Have a wonderful day'
]
test_sequences = tokenizer.texts_to_sequences(test_sentences)
padded_test_sequences = pad_sequences(test_sequences, maxlen=max_len, padding='post', truncating='post')
predictions = model.predict(padded_test_sequences)

# Decode the predictions to emoji strings
predicted_emoji_indices = [pred.argmax() + 1 for pred in predictions]  # Add 1 to match the tokenizer's index
predicted_emojis = emoji_tokenizer.sequences_to_texts([[idx] for idx in predicted_emoji_indices])
print(predicted_emojis)
