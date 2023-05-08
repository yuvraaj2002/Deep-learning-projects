import customtkinter
from customtkinter import *
import tensorflow as tf
from keras.preprocessing.text import one_hot
import numpy as np
from keras.utils import pad_sequences
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pycountry
import string

nltk.download('wordnet')
nltk.download('stopwords')
country_names = set([country.name for country in pycountry.countries])
lemm = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


def remove_punctuations(text):
    clean_text = [alphabet for alphabet in text if alphabet not in string.punctuation]
    clean_text = ''.join(clean_text)
    return clean_text


def lower_case(text):
    tokens_word = text.split()
    low = [word.lower() if word not in country_names else word for word in tokens_word]
    low = ' '.join(low)
    return low


def remove_stopwords(text):
    tokens_word = text.split()
    no_stop = [word for word in tokens_word if word not in stop_words]
    no_stop = ' '.join(no_stop)
    return no_stop


def lemmatize(text):
    tokens_word = text.split()
    lemm_word = [lemm.lemmatize(word) for word in tokens_word]
    lemm_word = ' '.join(lemm_word)
    return lemm_word


def text_prep(text):
    cleaned_text = remove_punctuations(text)
    cleaned_text = lower_case(cleaned_text)
    cleaned_text = remove_stopwords(cleaned_text)
    cleaned_text = lemmatize(cleaned_text)
    return cleaned_text


def predictor(text=''):
    corpus = [text_prep(text)]
    voc_size = 10000
    onehot_repr = [one_hot(words, voc_size) for words in corpus]
    sent_length = 30
    embedded_docs = pad_sequences(onehot_repr, padding='pre', maxlen=sent_length)
    x = np.array(embedded_docs)
    new_model = tf.keras.models.load_model('fake_news_detextor.h5')
    alp = new_model.predict(x)
    print(alp, alp[0], alp[0][0])
    if alp[0][0] >= 0.525:
        return f"Valid News Score: {alp[0][0]}"
    elif alp[0][0] < 0.525:
        return f"Fake News Score: {alp[0][0]}"


class App(customtkinter.CTk):
    APP_NAME = "ML Based Fake News Detector v1.0.0"
    WIDTH = 720
    HEIGHT = 405

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.title(App.APP_NAME)
        self.geometry(str(App.WIDTH) + "x" + str(App.HEIGHT))
        self.minsize(App.WIDTH, App.HEIGHT)
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.protocol("Return", self.print_var)
        self.createcommand('tk::mac::Quit', self.on_closing)
        self.headline = None

        self.grid_rowconfigure(0, weight=3)
        self.grid_rowconfigure(1, weight=10)
        self.grid_columnconfigure(0, weight=1)

        self.interaction_frame = customtkinter.CTkFrame(master=self, height=73, width=700)
        self.result_frame = customtkinter.CTkFrame(master=self, height=292, width=700)

        self.input_text_bex = customtkinter.CTkEntry(master=self.interaction_frame, placeholder_text="Enter Headline",
                                                     height=63, width=184, font=('Roboto', 20))
        self.show_result_button = customtkinter.CTkButton(master=self.interaction_frame, text="Analyse Headline",
                                                          height=63, width=36, command=self.print_var,
                                                          font=('Roboto', 20))
        self.result_box = customtkinter.CTkLabel(master=self.result_frame, text="Result", font=('Roboto', 20))

        self.set_frames()

    def set_frames(self):
        self.interaction_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        self.interaction_frame.grid_columnconfigure(0, weight=3)
        self.interaction_frame.grid_columnconfigure(1, weight=1)

        self.result_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

        self.input_text_bex.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        self.show_result_button.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")

        self.result_box.grid(row=0, column=0, padx=10, pady=10, sticky='nsew')

    def on_closing(self, event=0):
        self.destroy()
        exit()

    def print_var(self):
        self.headline = self.input_text_bex.get()
        self.result_box.configure(text="Input Headline: " + self.headline + "\nResult: " + predictor(self.input_text_bex.get()))
        self.input_text_bex.delete(0, END)

    def start(self):
        self.mainloop()


if __name__ == '__main__':
    predictor('Inside Ciara\'s Year of "Life and Love": Russell Wilson\'s Daddy Skills, Her Kids\' "Awesome" Bond and More')
    app = App()
    app.start()
