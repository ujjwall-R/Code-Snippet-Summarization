# Do unzip the python_dataset.zip first
datapath = 'Datasets/python_dataset.csv'
num_epochs = 10     # Set epochs
# Check 

import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pickle
from preprocess import preprocess, tokenize

df = pd.read_csv(datapath)
df = df[['code', 'docstring']]

### Preprocessing
print("Preprocessing...")
max_code_len = 100
max_summary_len = 25
df = preprocess(df, max_code_len, max_summary_len)

### Train-test split
X_train, X_val, Y_train, Y_val = train_test_split(
    np.array(df['code']),
    np.array(df['docstring']),
    test_size=0.15,
    random_state=0,
    shuffle=True
)

import os
if not os.path.exists('pickle'):
    os.makedirs('pickle')
    
### Tokenizing code
print("Tokenizing...code ")
x_tokenizer_path = 'pickle/x_tokenizer.pickle'
x_train = tokenize(list(X_train), max_pad_len=max_code_len, tokenizer_path=x_tokenizer_path, thresh=3, fit_on_texts=True)
with open(x_tokenizer_path, 'rb') as handle:
    x_tokenizer = pickle.load(handle)

x_vocab_size = x_tokenizer.num_words + 1

x_val = tokenize(list(X_val), max_pad_len=max_code_len, tokenizer_path=x_tokenizer_path, thresh=3, fit_on_texts=False)


### Tokenizing summary
print("Tokenizing...summary ")
y_tokenizer_path = 'pickle/y_tokenizer.pickle'
y_train = tokenize(list(Y_train), max_pad_len=max_summary_len, tokenizer_path=y_tokenizer_path, thresh=2, fit_on_texts=True)
with open(y_tokenizer_path, 'rb') as handle:
    y_tokenizer = pickle.load(handle)
    
y_vocab_size = y_tokenizer.num_words + 1

y_val = tokenize(list(Y_val), max_pad_len=max_summary_len, tokenizer_path=y_tokenizer_path, thresh=2, fit_on_texts=False)

# TODO: Using Glove


### Encoder Decoder Architecture
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Concatenate, TimeDistributed
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

latent_dim = 300
embedding_dim = 200

print("Building the model...")
### Encoder ###

encoder_inputs = Input(shape=(max_code_len, ))
# Embedding layer
enc_emb = Embedding(x_vocab_size, embedding_dim, trainable=True)(encoder_inputs)
# Encoder LSTM 1
encoder_lstm1 = LSTM(latent_dim, return_sequences=True, return_state=True, dropout=0.4, recurrent_dropout=0.4)
(encoder_output1, state_h1, state_c1) = encoder_lstm1(enc_emb)
# Encoder LSTM 2
encoder_lstm2 = LSTM(latent_dim, return_sequences=True, return_state=True, dropout=0.4, recurrent_dropout=0.4)
(encoder_output2, state_h2, state_c2) = encoder_lstm2(encoder_output1)
# Encoder LSTM 3
encoder_lstm3 = LSTM(latent_dim, return_state=True, return_sequences=True, dropout=0.4, recurrent_dropout=0.4)
(encoder_outputs, state_h, state_c) = encoder_lstm3(encoder_output2)

### Decoder ###
# Set up the decoder, using encoder_states as the initial state
decoder_inputs = Input(shape=(None, ))
# Embedding layer
dec_emb_layer = Embedding(y_vocab_size, embedding_dim, trainable=True)
dec_emb = dec_emb_layer(decoder_inputs)
# Decoder LSTM
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True, dropout=0.4, recurrent_dropout=0.2)
(decoder_outputs, decoder_fwd_state, decoder_back_state) = decoder_lstm(dec_emb, initial_state=[state_h, state_c])
## Dense layer
decoder_dense = TimeDistributed(Dense(y_vocab_size, activation='softmax'))
decoder_outputs = decoder_dense(decoder_outputs)

## Define the model
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
# model.summary()
model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy')


# Early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=2)
# ModelCheckpoint callback to save the best model during training
checkpoint_path = 'model_checkpoints/model.{epoch:02d}-{val_loss:.2f}.h5'
checkpoint_callback = ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True)


### Training
print("Training the model...")
history = model.fit(
    [x_train, y_train[:, :-1]],
    y_train.reshape(y_train.shape[0], y_train.shape[1], 1)[:, 1:],
    epochs=num_epochs,
    batch_size=128,
    validation_data=([x_val, y_val[:, :-1]],
                     y_val.reshape(y_val.shape[0], y_val.shape[1], 1)[:, 1:]),
    callbacks=[early_stopping, checkpoint_callback]
)
# TODO: Experiment with various configurations and hyperparameters like batch_size, epoch, etc.


model.save('trained_model.h5')

plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='val')
plt.legend()
plt.savefig('loss_plot.png')


### Define prediction models
reverse_target_word_index = y_tokenizer.index_word
reverse_source_word_index = x_tokenizer.index_word
target_word_index = y_tokenizer.word_index

## Encoder
encoder_model = Model(inputs=encoder_inputs, outputs=[encoder_outputs, state_h, state_c])

## Decoder
# Tensors to hold the states of the previous time step
decoder_state_input_h = Input(shape=(latent_dim, ))
decoder_state_input_c = Input(shape=(latent_dim, ))
decoder_hidden_state_input = Input(shape=(max_code_len, latent_dim))  # encoded code sequence

# Get the embeddings of the decoder sequence
dec_emb2 = dec_emb_layer(decoder_inputs) 
# To predict the next word in the sequence, set the initial states to the states from the previous time step
(decoder_outputs2, state_h2, state_c2) = decoder_lstm(dec_emb2,
        initial_state=[decoder_state_input_h, decoder_state_input_c])
# A dense softmax layer to generate prob dist. over the target vocabulary
decoder_outputs2 = decoder_dense(decoder_outputs2)
# Final decoder model
decoder_model = Model([decoder_inputs] + [decoder_hidden_state_input,
                      decoder_state_input_h, decoder_state_input_c],
                      [decoder_outputs2] + [state_h2, state_c2])

encoder_model.save('encoder_model.h5')
decoder_model.save('decoder_model.h5')