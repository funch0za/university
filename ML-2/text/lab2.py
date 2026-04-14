import torch
import pandas as pd
import numpy as np
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import TensorDataset

data = pd.read_csv("r_preprocessed.csv")

all_words = " ".join(data.processed.values).split()
#print(all_words)

counter = Counter(all_words)

print(counter)
vocabulary = sorted(counter, key=counter.get, reverse=True)
#print(len(vocabulary))

int2word = dict(enumerate(vocabulary, 1)) # присваивается ID
int2word[0] = "PAD"

word2int = {word: id for id, word in int2word.items()} # (слово, ID)

# кодирование текстов в числа
reviews = data.processed.values
all_words = " ".join(reviews).split()

# каждый отзыв разбивается на слова, и каждое слово заменяется на его числовой ID из словаря word2int
review_enc = [[word2int[word] for word in review.split()] for review in reviews]

# padding
sequence_length = 256
# приводим отзывы к длине из 256 токенов
reviews_padding = np.full((len(review_enc), sequence_length), word2int['PAD'], dtype=int)

# создание матрицы
for i, row in enumerate(review_enc):
    reviews_padding[i, :len(row)] = np.array(row)[:sequence_length]

# метки классов
labels = data.label.to_numpy()
train_len = 0.6 # 60% на обучение
test_len = 0.5 # от остатка 50% на тест, 50% на валидацию
train_last_index = int(len(reviews_padding) * train_len)

train_x, remainder_x = reviews_padding[:train_last_index], reviews_padding[train_last_index: ]
train_y, remainder_y = labels[:train_last_index], labels[train_last_index:]

test_last_index = int(len(remainder_x)*test_len)
test_x = remainder_x[:test_last_index]
test_y = remainder_y[:test_last_index]

check_x = remainder_x[test_last_index:]
check_y = remainder_y[test_last_index:]

train_dataset = TensorDataset( torch.from_numpy(train_x), torch.from_numpy(train_y))
test_dataset = TensorDataset(torch.from_numpy(test_x), torch. from_numpy(test_y))
check_dataset = TensorDataset(torch. from_numpy(check_x), torch.from_numpy(check_y))

batch_size = 128
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
test_loader = DataLoader(test_dataset, shuffle=True, batch_size=batch_size)
check_loader = DataLoader(check_dataset, shuffle=True, batch_size=batch_size)

