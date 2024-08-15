import streamlit as st
import pandas as pd
import numpy as np
from textblob import TextBlob
from googletrans import Translator
import re
import string
from nltk.tokenize import word_tokenize
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import matplotlib.pyplot as plt
import time

#ANALISIS SENTIMEN
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from collections import defaultdict

# NLP
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
import emoji

# Viz
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl

#Model IndoBERT
import random
import torch
import torch.optim as optim

import torch.nn.functional as F
from torch import optim
from tqdm import tqdm

from transformers import BertForSequenceClassification, BertConfig, BertTokenizer
from indonlu.utils.data_utils import DocumentSentimentDataset, DocumentSentimentDataLoader
from indonlu.utils.forward_fn import forward_sequence_classification
from indonlu.utils.metrics import document_sentiment_metrics_fn
from streamlit_option_menu import option_menu
# Initialize the stopword remover and stemmer
stopword_factory = StopWordRemoverFactory()
stopword_remover = stopword_factory.create_stop_word_remover()
stemmer_factory = StemmerFactory()
stemmer = stemmer_factory.create_stemmer()


# Function to translate text to English and calculate sentiment
st.markdown("""
        <style>
            .stApp {
                background-color: lightblue;
        }
        </style>
     """, unsafe_allow_html=True)

def translate_and_calculate_sentiment(df, column):
    translator = Translator()
    df['translated_text'] = df[column].apply(lambda x: translator.translate(x, src='id', dest='en').text)
    df['polarity'] = df['translated_text'].apply(lambda x: TextBlob(x).sentiment.polarity)
    df['sentimen'] = df['polarity'].apply(lambda x: 'positive' if x > 0 else ('negative' if x < 0 else 'neutral'))
    return df[['judul', 'sentimen']]

# Function to preprocess data
def preprocessing_data(df):
    df['case_folding'] = df['judul'].apply(lambda x: x.lower())
    
    def clean_text(text):
        text = text.replace('\\t', " ").replace('\\n'," ").replace('\\u', " ").replace('\\',"")
        text = text.encode('ascii', 'replace').decode('ascii')
        text = ' '.join(re.sub(r"(\w+:\/\/\s+)", " ", text).split())
        text = re.sub(r"\d+","", text)
        text = text.replace('-',' ')
        text = text.translate(str.maketrans("","", string.punctuation))
        text = re.sub(r'\s+',' ', str(text))
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    df['cleaning'] = df['case_folding'].apply(clean_text)

    def word_tokenize_wrapper(text):
        return word_tokenize(text)

    df['tokenize'] = df['cleaning'].apply(word_tokenize_wrapper)

    def remove_stopwords(tokens):
        text = ' '.join(tokens)
        text = stopword_remover.remove(text)
        return text.split()

    df['stopwords_removal'] = df['tokenize'].apply(remove_stopwords)

    term_dict = {}

    for document in df['stopwords_removal']:
        for term in document:
            if term not in term_dict:
                term_dict[term] = ' '

    for term in term_dict:
        term_dict[term] = stemmer.stem(term)

    def apply_stemming(document):
        return [term_dict[term] for term in document]
    df['stemming'] = df['stopwords_removal'].apply(apply_stemming)
    df['clean'] = df['stemming'].apply(lambda x: ' '.join(x))
    df = df[['clean', 'sentimen']]
    df = df.rename(columns={'clean': 'judul'})

    return df.reset_index(drop=True)

#KODING UNTUNG ANALISIS SENTIMEN
# Function to make a donut chart
def Analisis_Sentimen(df):
    fig, ax = plt.subplots()
    sizes = df['sentimen'].value_counts()
    labels = ['Negative', 'Neutral', 'Positive']
    colors = ['lightcoral', 'lightskyblue', 'lightgreen']
    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
    ax.axis('equal')
    st.pyplot(fig)

    """Tahap Training Dataset"""
    # train val split
    train_set, val_set = train_test_split(df, test_size=0.3, stratify=df.sentimen, random_state=1)
    val_set, test_set = train_test_split(val_set, test_size=0.33, stratify=val_set.sentimen, random_state=1)

    print(f'Train shape: {train_set.shape}')
    print(f'Val shape: {val_set.shape}')
    print(f'Test shape: {test_set.shape}')

    # export to tsv
    train_set.to_csv('train_set.tsv', sep='\t', header=None, index=False)
    val_set.to_csv('val_set.tsv', sep='\t', header=None, index=False)
    test_set.to_csv('test_set.tsv', sep='\t', header=None, index=False)

    # common functions
    def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    def count_param(module, trainable=False):
        if trainable:
            return sum(p.numel() for p in module.parameters() if p.requires_grad)
        else:
            return sum(p.numel() for p in module.parameters())

    def get_lr(optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    def metrics_to_string(metric_dict):
        string_list = []
        for key, value in metric_dict.items():
            string_list.append('{}:{:.2f}'.format(key, value))
        return ' '.join(string_list)

    # Set random seed
    set_seed(27)

    # Load Tokenizer and Config
    tokenizer = BertTokenizer.from_pretrained('indobenchmark/indobert-base-p1')
    config = BertConfig.from_pretrained('indobenchmark/indobert-base-p1')
    config.num_labels = DocumentSentimentDataset.NUM_LABELS

    # Instantiate model
    model = BertForSequenceClassification.from_pretrained('indobenchmark/indobert-base-p1', config=config)

    # Jumlah parameter
    print(count_param(model))

    train_dataset_path = 'train_set.tsv'
    valid_dataset_path = 'val_set.tsv'
    test_dataset_path = 'test_set.tsv'

    # fungsi dataset loader dari utils IndoNLU
    train_dataset = DocumentSentimentDataset(train_dataset_path, tokenizer, lowercase=True)
    valid_dataset = DocumentSentimentDataset(valid_dataset_path, tokenizer, lowercase=True)
    test_dataset = DocumentSentimentDataset(test_dataset_path, tokenizer, lowercase=True)

    train_loader = DocumentSentimentDataLoader(dataset=train_dataset, max_seq_len=512, batch_size=32, num_workers=2, shuffle=True, multiprocessing_context="spawn")
    valid_loader = DocumentSentimentDataLoader(dataset=valid_dataset, max_seq_len=512, batch_size=32, num_workers=2, shuffle=False)
    test_loader = DocumentSentimentDataLoader(dataset=test_dataset, max_seq_len=512, batch_size=32, num_workers=2, shuffle=False)

    w2i, i2w = DocumentSentimentDataset.LABEL2INDEX, DocumentSentimentDataset.INDEX2LABEL
    print(w2i) # word to index
    print(i2w) # index to word

    # Tentukan optimizer
    optimizer = optim.Adam(model.parameters(), lr=3e-6)

    # Train
    n_epochs = 5
    history = defaultdict(list)
    for epoch in range(n_epochs):
        model.train()
        torch.set_grad_enabled(True)

        total_train_loss = 0
        list_hyp_train, list_label = [], []

        train_pbar = tqdm(train_loader, leave=True, total=len(train_loader))
        for i, batch_data in enumerate(train_pbar):
            # Forward model
            loss, batch_hyp, batch_label = forward_sequence_classification(model, batch_data[:-1], i2w=i2w, device='cpu')

            # Update model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            tr_loss = loss.item()
            total_train_loss = total_train_loss + tr_loss

            # Hitung skor train metrics
            list_hyp_train += batch_hyp
            list_label += batch_label

            train_pbar.set_description("(Epoch {}) TRAIN LOSS:{:.4f} LR:{:.8f}".format((epoch+1),
                total_train_loss/(i+1), get_lr(optimizer)))

        metrics = document_sentiment_metrics_fn(list_hyp_train, list_label)
        print("(Epoch {}) TRAIN LOSS:{:.4f} {} LR:{:.8f}".format((epoch+1),
            total_train_loss/(i+1), metrics_to_string(metrics), get_lr(optimizer)))

        # save train acc for learning curve
        history['train_acc'].append(metrics['ACC'])

        # Evaluate di validation set
        model.eval()
        torch.set_grad_enabled(False)

        total_loss, total_correct, total_labels = 0, 0, 0
        list_hyp, list_label = [], []

        pbar = tqdm(valid_loader, leave=True, total=len(valid_loader))
        for i, batch_data in enumerate(pbar):
            batch_seq = batch_data[-1]
            loss, batch_hyp, batch_label = forward_sequence_classification(model, batch_data[:-1], i2w=i2w, device='cpu')

            # Hitung total loss
            valid_loss = loss.item()
            total_loss = total_loss + valid_loss

            # Hitung skor evaluation metrics
            list_hyp += batch_hyp
            list_label += batch_label
            metrics = document_sentiment_metrics_fn(list_hyp, list_label)

            pbar.set_description("VALID LOSS:{:.4f} {}".format(total_loss/(i+1), metrics_to_string(metrics)))

        metrics = document_sentiment_metrics_fn(list_hyp, list_label)
        print("(Epoch {}) VALID LOSS:{:.4f} {}".format((epoch+1),
            total_loss/(i+1), metrics_to_string(metrics)))

        # save validation acc for learning curve
        history['val_acc'].append(metrics['ACC'])

    # Plot learning curve
    fig1, ax = plt.subplots()
    plt.plot(history['train_acc'], label='train acc')
    plt.plot(history['val_acc'], label='validation acc')
    plt.title('Training history')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.ylim([0, 1])
    st.pyplot(fig1)

    # Simpan hasil validasi
    val_df = pd.read_csv(valid_dataset_path, sep='\t', names=['judul', 'sentimen'])
    val_df['pred'] = list_hyp
    st.write("Validation Data with Predictions:")
    st.write(val_df)
    val_df.to_csv('val hasil.csv', index=False)

    # Prediksi test set
    model.eval()
    torch.set_grad_enabled(False)
    pred = []

    pbar = tqdm(test_loader, leave=True, total=len(test_loader))
    for i, batch_data in enumerate(pbar):
        _, batch_hyp, _ = forward_sequence_classification(model, batch_data[:-1], i2w=i2w, device='cpu')
        pred += batch_hyp

    # Simpan hasil prediksi test set
    test_df = pd.read_csv(test_dataset_path, sep='\t', names=['judul', 'sentimen'])
    test_df['pred'] = pred
    # Tampilkan hasil prediksi test set di aplikasi Streamlit
    st.subheader("Test Data with Predictions")
    st.write(test_df)
    # Simpan ke file CSV
    test_df.to_csv('test_result.csv', index=False)
    test_df.head()

    

# Define functions to display confusion matrix and classification report
    def show_confusion_matrix(confusion_matrix):
        fig3, ax2 = plt.subplots()
        hmap = sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues")
        hmap.yaxis.set_ticklabels(hmap.yaxis.get_ticklabels(), rotation=0, ha='right')
        hmap.xaxis.set_ticklabels(hmap.xaxis.get_ticklabels(), rotation=30, ha='right')
        plt.ylabel('True Sentiment')
        plt.xlabel('Predicted Sentiment');
        st.pyplot(fig3)

    def display_classification_report(y_true, y_pred):
        report = classification_report(y_true, y_pred, target_names=['Negative', 'Neutral', 'Positive'])
        st.text(report)

# Calculate confusion matrix and classification report for validation set
    val_real = val_df['sentimen']
    val_pred = val_df['pred']

    cm_val = confusion_matrix(val_real, val_pred)
    df_cm_val = pd.DataFrame(cm_val, index=['Negative', 'Neutral', 'positive'], columns=['Negative', 'Neutral', 'positive'])

    st.write("Validation Set:")
    st.write("Confusion Matrix:")
    show_confusion_matrix(df_cm_val)

    st.write("Classification Report:")
    display_classification_report(val_real, val_pred)

    # Calculate confusion matrix and classification report for test set
    test_real = test_df['sentimen']
    test_pred = test_df['pred']

    cm_test = confusion_matrix(test_real, test_pred)
    df_cm_test = pd.DataFrame(cm_test, index=['Negative', 'Neutral', 'positive'], columns=['Negative', 'Neutral', 'positive'])

    st.write("Test Set:")
    st.write("Confusion Matrix:")
    show_confusion_matrix(df_cm_test)

    st.write("Classification Report:")
    display_classification_report(test_real, test_pred)

    # Function for home page content
import streamlit as st

# CSS untuk mengubah warna tombol menu dan aplikasi


# Fungsi halaman beranda

# Define the home page function
def home_page():
    st.markdown("<div style='text-align: center'>", unsafe_allow_html=True)
    st.title("SELAMAT DATANG DI APLIKASI ANALISIS SENTIMEN")
    st.markdown("""
    <div style='font-family: 'Rubik', sans-serif;'>
    
    ### Cara Penggunaan
    
    1. **Labelisasi**:
       - Unggah file CSV yang berisi teks berbahasa Indonesia.
       - Pilih opsi untuk menerjemahkan dan menganalisis sentimen teks.
    
    2. **Preprocessing Data**:
       - Gunakan data yang sudah diterjemahkan untuk diproses sebelum dilakukan analisis sentimen.
    
    3. **Analisis Sentimen**:
       - Analisis sentimen pada data yang sudah diproses dan tampilkan hasilnya.
    
    </div>
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


# Sidebar menu
with st.sidebar:
    st.markdown("<style> @import url('https://fonts.googleapis.com/css2?family=Rubik:ital,wght@0,300..900;1,300..900&display=swap'); </style>", unsafe_allow_html=True)
    selection = option_menu("SENTIMEN IndoBERT", ["Home", "Labelisasi", "Preprocessing Data", "Analisis Sentimen"], 
                            icons=['house', 'pencil-square', 'hourglass', 'bar-chart-line'], menu_icon="cast", default_index=0)


# Home Page
if selection == "Home":
    home_page()
    
    

# Labelisasi Page
elif selection == "Labelisasi":
    st.subheader("📝Labelisasi")
    uploaded_file_translate = st.file_uploader("Pilih file CSV", type="csv")
    if uploaded_file_translate is not None:
        df_translate = pd.read_csv(uploaded_file_translate)
        st.write("Dataset yang diunggah:")
        
        st.write(df_translate)
        if st.button("Labelisasi"):
            if 'judul' in df_translate.columns:
                
                df_translated = translate_and_calculate_sentiment(df_translate, 'judul')
                st.write("Data setelah diterjemahkan dan dianalisis sentimen:")
                st.write(df_translated)
                st.session_state.df_translated = df_translated.copy()
            else:
                st.write("Format CSV tidak valid. Pastikan CSV Anda mengandung kolom 'judul'.")

# Preprocess Data Page
elif selection == "Preprocessing Data":
    st.subheader("🔄Preprocessing Data")
    
    if 'df_translated' in st.session_state and st.session_state.df_translated is not None:
        st.write("Menggunakan data yang sudah diterjemahkan sebelumnya.")
        st.write("Data yang sudah diterjemahkan:")
        st.write(st.session_state.df_translated)
        if st.button("Proses Data"):
            
            df_preprocessed = preprocessing_data(st.session_state.df_translated)
            st.write("Data setelah diproses:")
            st.write(df_preprocessed)
            st.session_state.df_preprocessed = df_preprocessed.copy()
            st.session_state.df_translated = None
        
    else:
        st.write("Unggah file CSV untuk diproses.")
        uploaded_file_preprocess = st.file_uploader("Pilih file CSV", type="csv")
        if uploaded_file_preprocess is not None:
            df_preprocess = pd.read_csv(uploaded_file_preprocess)
            st.write("Dataset yang diunggah:")
            st.write(df_preprocess)
            if st.button("Proses Data"):
                if 'judul' in df_preprocess.columns:
                    
                    df_preprocessed = preprocessing_data(df_preprocess)
                    st.write("Data setelah diproses:")
                    st.write(df_preprocessed)
                    st.session_state.df_preprocessed = df_preprocessed.copy()
                else:
                    st.write("Format CSV tidak valid. Pastikan CSV Anda mengandung kolom 'judul'.")

# Analisis Sentimen Page
elif selection == "Analisis Sentimen":
    st.subheader("📊Analisis Sentimen")
    
    if 'df_preprocessed' in st.session_state and st.session_state.df_preprocessed is not None:
        st.write("Menggunakan data yang sudah diproses sebelumnya.")
        st.write("Data yang sudah diproses:")
        st.write(st.session_state.df_preprocessed)
        if st.button("Analisis Sentimen"):
           
            df_sentimen = Analisis_Sentimen(st.session_state.df_preprocessed)
            if df_sentimen is not None:
                st.write("Data setelah analisis sentimen:")
                st.write(df_sentimen)
                st.session_state.df_sentimen = df_sentimen.copy()
                st.session_state.df_preprocessed = None
    else:
        st.write("Unggah file CSV untuk dianalisis.")
        uploaded_file_sentimen = st.file_uploader("Pilih file CSV", type="csv")
        if uploaded_file_sentimen is not None:
            df_preprocess = pd.read_csv(uploaded_file_sentimen)
            st.write("Dataset yang diunggah:")
            st.write(df_preprocess)
            if st.button("Analisis Sentimen"):
                if 'judul' in df_preprocess.columns:
                   
                    df_sentimen = Analisis_Sentimen(df_preprocess)
                    st.write("Data setelah analisis sentimen:")
                    st.write(df_sentimen)
                    st.session_state.df_sentimen = df_sentimen.copy()
                else:
                    st.write("Format CSV tidak valid. Pastikan CSV Anda mengandung kolom 'judul'.")