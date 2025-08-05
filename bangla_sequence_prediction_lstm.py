import numpy as np
import pandas as pd
import re
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import pickle
import tkinter as tk
from tkinter import messagebox

### ---- Data Preprocessing Functions ---- ###

def clean_text(text):
    # Remove punctuation, numbers, other unicode symbols, and extra spaces
    text = re.sub(r'[^\u0980-\u09FF\s]', '', text)  # Bengali unicode range
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def remove_duplicates(sentences):
    return list(set(sentences))

def remove_stopwords(sentence, stopwords):
    return ' '.join([word for word in sentence.split() if word not in stopwords])

def stem_bengali_word(word):
    # Simple stemming: remove common Bengali suffixes (could be improved)
    suffixes = ['রা', 'দের', 'র', 'ের', 'েরা', 'ে', 'রা', 'টি', 'টা', 'গুলো', 'খানা', 'গুলি', 'টা']
    for suf in suffixes:
        if word.endswith(suf):
            return word[:-len(suf)]
    return word

def stem_sentences(sentences):
    stemmed_sentences = []
    for sentence in sentences:
        stemmed = ' '.join([stem_bengali_word(word) for word in sentence.split()])
        stemmed_sentences.append(stemmed)
    return stemmed_sentences

def remove_small_docs(sentences, min_length=5):
    return [s for s in sentences if len(s.split()) >= min_length]

### ---- Load and Preprocess Data ---- ###

# Assume data is a CSV with columns: 'article', 'category'
DATA_PATH = 'ittefaq_articles.csv' # Update with your actual path

def load_and_preprocess():
    df = pd.read_csv(DATA_PATH)
    # Combine all articles into one list of sentences
    sentences = []
    for doc in df['article'].dropna().values:
        doc = clean_text(doc)
        sentences.extend([s for s in doc.split('।') if len(s.strip()) > 0])  # Bengali full stop
    sentences = remove_duplicates(sentences)
    # Load or define a list of Bengali stopwords
    stopwords = set(['ও', 'এবং', 'আর', 'কিন্তু', 'যে', 'তবে', 'তাতে', 'সে', 'এই', 'তো', 'তার', 'তাদের', 'যা', 'যাতে', 'হয়', 'গিয়েছে', 'গেল', 'দিয়ে', 'ছিল', 'করে', 'থেকে', 'উপর', 'নিয়ে', 'জন্য', 'সঙ্গে', 'পর', 'আমরা', 'তুমি', 'তোমরা', 'আপনি', 'আপনার', 'তাদের', 'কেন', 'কি', 'কী'])
    sentences = [remove_stopwords(s, stopwords) for s in sentences]
    sentences = stem_sentences(sentences)
    sentences = remove_small_docs(sentences)
    return sentences

### ---- Sequence Preparation ---- ###

def create_sequences(sentences, tokenizer, max_sequence_len):
    input_sequences = []
    for line in sentences:
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(2, len(token_list)):
            n_gram_sequence = token_list[:i+1]
            input_sequences.append(n_gram_sequence)
    input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))
    predictors, label = input_sequences[:,:-1], input_sequences[:,-1]
    label = tf.keras.utils.to_categorical(label, num_classes=tokenizer.num_words)
    return predictors, label

### ---- Model Building ---- ###

def build_lstm_model(max_sequence_len, total_words, embedding_dim=128, lstm_units=256):
    model = Sequential()
    model.add(Embedding(input_dim=total_words, output_dim=embedding_dim, input_length=max_sequence_len-1))
    model.add(LSTM(lstm_units))
    model.add(Dropout(0.2))
    model.add(Dense(total_words, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

### ---- Train Model ---- ###

def train_and_save_model():
    sentences = load_and_preprocess()
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(sentences)
    total_words = len(tokenizer.word_index) + 1
    max_sequence_len = max([len(x.split()) for x in sentences]) + 1
    predictors, label = create_sequences(sentences, tokenizer, max_sequence_len)
    model = build_lstm_model(max_sequence_len, total_words)
    es = EarlyStopping(monitor='loss', patience=3, restore_best_weights=True)
    history = model.fit(predictors, label, epochs=20, batch_size=512, callbacks=[es])
    # Save model and tokenizer
    model.save('bangla_lstm_seq_model.h5')
    with open('bangla_tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)
    with open('bangla_seq_params.pkl', 'wb') as f:
        pickle.dump({'max_sequence_len': max_sequence_len, 'total_words': total_words}, f)
    # Plot training accuracy
    plt.plot(history.history['accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.show()

### ---- Prediction Function ---- ###

def load_model_and_tokenizer():
    model = tf.keras.models.load_model('bangla_lstm_seq_model.h5')
    with open('bangla_tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    with open('bangla_seq_params.pkl', 'rb') as f:
        params = pickle.load(f)
    return model, tokenizer, params

def predict_next_words(seed_text, model, tokenizer, max_sequence_len, n_words=1):
    for _ in range(n_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted = model.predict(token_list, verbose=0)
        predicted_word_index = np.argmax(predicted, axis=1)[0]
        output_word = ''
        for word, index in tokenizer.word_index.items():
            if index == predicted_word_index:
                output_word = word
                break
        seed_text += " " + output_word
    return seed_text

### ---- Tkinter UI ---- ###

def launch_ui():
    model, tokenizer, params = load_model_and_tokenizer()
    max_sequence_len = params['max_sequence_len']

    def on_predict():
        seed = entry.get().strip()
        if not seed:
            messagebox.showerror("Error", "Please enter a Bangla text seed.")
            return
        try:
            result = predict_next_words(seed, model, tokenizer, max_sequence_len, n_words=3)
            output_var.set(result)
        except Exception as e:
            messagebox.showerror("Prediction Error", str(e))

    root = tk.Tk()
    root.title("Bangla Sequence Prediction using LSTMs")
    root.geometry("500x250")
    tk.Label(root, text="Enter Bangla Seed Text:", font=('Arial', 12)).pack(pady=10)
    entry = tk.Entry(root, width=50, font=('Arial', 14))
    entry.pack(pady=5)
    tk.Button(root, text="Predict Next Words", command=on_predict, font=('Arial', 12)).pack(pady=10)
    output_var = tk.StringVar()
    tk.Label(root, textvariable=output_var, wraplength=400, font=('Arial', 14), fg="green").pack(pady=10)
    root.mainloop()

### ---- Main Entrypoint ---- ###

if __name__ == "__main__":
    # To train and save model, uncomment the following line:
    # train_and_save_model()
    # To run the UI, first ensure you have trained and saved the model.
    launch_ui()