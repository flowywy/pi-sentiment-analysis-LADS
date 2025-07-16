import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
import streamlit as st

# Membuat dan menampilkan wordcloud di Streamlit
def generate_wordcloud(text_series, title):
    text = " ".join(text_series)
    wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='magma').generate(text)
    st.subheader(f"{title}")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    st.pyplot(fig)

# Menyimpan wordcloud ke file image
def save_wordcloud_image(text_series, filename):
    text = " ".join(text_series)
    wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='magma').generate(text)
    wordcloud.to_file(filename)

# Mendapatkan top n-grams (misalnya trigram)
def get_top_ngrams(text_series, ngram_range=(3,3), n=10):
    vectorizer = CountVectorizer(stop_words='english', ngram_range=ngram_range)
    X = vectorizer.fit_transform(text_series)
    counts = X.toarray().sum(axis=0)
    ngrams_freq = dict(zip(vectorizer.get_feature_names_out(), counts))
    top_ngrams = sorted(ngrams_freq.items(), key=lambda x: x[1], reverse=True)[:n]
    return top_ngrams

# Membuat bar chart horizontal untuk top n-grams
def plot_top_ngrams_bar_chart(top_ngrams, title):
    if top_ngrams:
        ngrams, counts = zip(*top_ngrams)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.barh(ngrams, counts, color='#7F55B1')
        ax.invert_yaxis()  # agar n-gram frekuensi tertinggi di atas
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('Frequency')
        ax.set_ylabel('N-gram')
        st.pyplot(fig)

# Membuat dan menyimpan summary chart (bar chart jumlah positif & negatif)
def save_summary_chart(df, filename):
    count_positive = (df['Sentiment'] == 'positive').sum()
    count_negative = (df['Sentiment'] == 'negative').sum()
    summary_df = pd.DataFrame({
        'Sentiment': ['Positive', 'Negative'],
        'Count': [count_positive, count_negative]
    })
    fig, ax = plt.subplots()
    ax.bar(summary_df['Sentiment'], summary_df['Count'], color=['green', 'red'])
    ax.set_title("Sentiment Summary")
    fig.savefig(filename)
    plt.close(fig)
