import numpy as np
from tensorflow.keras.layers import Dense, LSTM, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

add_column_text = ["add a column","insert a column","place new column","add row numbers","insert row numbers"]
remove_column_text = ["remove column","delete column", "remove row numbers"]

augmented_add_column = []
for txt in add_column_text:
  for num in range(9):
    augmented_add_column.append(f'{txt} after column {num + 1}')
    augmented_add_column.append(f'{txt} before column {num + 1}')

augmented_remove_column = []
for txt in remove_column_text:
  for num in range(9):
    augmented_remove_column.append(f'{txt} after column {num + 1}')
    augmented_remove_column.append(f'{txt} before column {num + 1}')

dict = {}
dict['<unknown>'] = 0
dict['a'] = 1
dict['add'] = 2
dict['column'] = 3
dict['delete'] = 4
dict['insert'] = 5
dict['new'] = 6
dict['numbers'] = 7
dict['place'] = 8
dict['remove'] = 9
dict['row'] = 10

seq_length = 6
rnn_window = 3


def text_to_one_hot(text, indices):
  text_arr = text.split()
  text_arr += [''] * (seq_length - len(text_arr))
  arr = np.zeros((len(indices),seq_length))
  for idx, token in enumerate(text_arr):
    if token in indices:
      oh_idx = indices[token]
    else:
      oh_idx = 0
    arr[oh_idx][idx] = 1.0
  return arr

def build_lstm_model():
  input = Input(shape=(len(dict), seq_length))
  x = LSTM(rnn_window, return_sequences=True)(input)
  x = LSTM(rnn_window, return_sequences=True)(x)
  x = LSTM(rnn_window)(x)
  x = Dense(len(dict), activation='softmax')(x)
  return Model(input, x)

# Transformations selected:
# [add_column, remove_column, unknown]
add_column_classification = np.array([[1.0], [0.0], [0.0]])
remove_column_classification = np.array([[0.0], [1.0], [0.0]])

add_column_y = np.full((len(augmented_add_column), 3, 1), add_column_classification)
remove_column_y = np.full((len(augmented_remove_column), 3, 1), remove_column_classification)
y = np.concatenate((add_column_y, remove_column_y), axis=0)

inputs = []
for text in augmented_add_column:
  input = text_to_one_hot(text, dict)
  inputs.append(input)

for text in augmented_remove_column:
  input = text_to_one_hot(text, dict)
  inputs.append(input)

X = np.array(inputs)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10)
