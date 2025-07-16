from fpdf import FPDF
from datetime import datetime
import pandas as pd
import re
import matplotlib.pyplot as plt
import os
from sklearn.feature_extraction.text import CountVectorizer

# Fungsi hapus emoji atau karakter non-ASCII
def remove_emojis(text):
    return re.sub(r'[^\x00-\x7F]+', '', text)

# Fungsi buat top n-gram chart & simpan ke file gambar
def save_top_ngrams_chart(text_series, sentiment_label, ngram_range=(3,3), n=5):
    vectorizer = CountVectorizer(stop_words='english', ngram_range=ngram_range)
    X = vectorizer.fit_transform(text_series)
    counts = X.toarray().sum(axis=0)
    ngrams_freq = dict(zip(vectorizer.get_feature_names_out(), counts))
    top_ngrams = sorted(ngrams_freq.items(), key=lambda x: x[1], reverse=True)[:n]

    if top_ngrams:
        ngrams, freqs = zip(*top_ngrams)
        plt.figure(figsize=(8, 4))
        plt.barh(ngrams, freqs, color='#7F55B1')
        plt.xlabel("Frequency")
        plt.title(f"Top Trigrams - {sentiment_label}")
        plt.gca().invert_yaxis()
        if not os.path.exists("temp"):
            os.makedirs("temp")
        filename = f"temp/top_ngrams_{sentiment_label.lower()}.png"
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
        return filename
    return None

# Membuat summary PDF
def create_summary_pdf(df, count_positive, count_negative, percent_positive, percent_negative, positive_texts_exist, negative_texts_exist):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Header
    pdf.set_font("Arial", "B", 18)
    pdf.cell(0, 10, "Sentiment Analysis Of Love And Deepspace", ln=True, align='C')
    pdf.ln(5)

    total = count_positive + count_negative

    # Average text length
    df['TextLength'] = df['content'].astype(str).apply(lambda x: len(x.split()))
    avg_length_positive = df[df['Sentiment'] == 'positive']['TextLength'].mean()
    avg_length_negative = df[df['Sentiment'] == 'negative']['TextLength'].mean()

    # Date generated
    date_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Ringkasan statistik
    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 8,
        f"Total data: {total}\n"
        f"Positive: {count_positive} ({percent_positive}%)\n"
        f"Negative: {count_negative} ({percent_negative}%)\n"
        f"Average text length (positive): {avg_length_positive:.1f} words\n"
        f"Average text length (negative): {avg_length_negative:.1f} words\n"
        f"Date generated: {date_now}"
    )
    pdf.ln(5)

    # Line separator
    pdf.set_draw_color(200, 200, 200)
    pdf.set_line_width(0.5)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(5)

    # Chart summary
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Sentiment Summary Chart", ln=True)
    pdf.image("temp/summary_chart.png", w=180)
    pdf.ln(10)

    # Wordcloud Positive
    if positive_texts_exist:
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "Wordcloud Positive", ln=True)
        pdf.image("temp/wordcloud_positive.png", w=180)
        pdf.ln(10)

    # Wordcloud Negative
    if negative_texts_exist:
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "Wordcloud Negative", ln=True)
        pdf.image("temp/wordcloud_negative.png", w=180)
        pdf.ln(10)

    # Example Positive Texts
    if positive_texts_exist:
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "Top 3 Positive Example Texts", ln=True)
        pdf.set_font("Arial", "", 12)
        pos_examples = df[df['Sentiment'] == 'positive']['content'].head(3).tolist()
        for i, text in enumerate(pos_examples, 1):
            clean_text = remove_emojis(text[:500].replace("\n", " "))
            pdf.multi_cell(0, 8, f"{i}. {clean_text}")
            pdf.ln(2)
        pdf.ln(5)

    # Example Negative Texts
    if negative_texts_exist:
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "Top 3 Negative Example Texts", ln=True)
        pdf.set_font("Arial", "", 12)
        neg_examples = df[df['Sentiment'] == 'negative']['content'].head(3).tolist()
        for i, text in enumerate(neg_examples, 1):
            clean_text = remove_emojis(text[:500].replace("\n", " "))
            pdf.multi_cell(0, 8, f"{i}. {clean_text}")
            pdf.ln(2)
        pdf.ln(5)

    # Top n-grams Positive 
    if positive_texts_exist:
        pos_filename = save_top_ngrams_chart(df[df['Sentiment'] == 'positive']['content'], "Positive")
        if pos_filename:
            pdf.set_font("Arial", "B", 14)
            pdf.cell(0, 10, "Top Trigrams - Positive", ln=True)
            pdf.image(pos_filename, w=180)
            pdf.ln(10)

    # Top n-grams Negative 
    if negative_texts_exist:
        neg_filename = save_top_ngrams_chart(df[df['Sentiment'] == 'negative']['content'], "Negative")
        if neg_filename:
            pdf.set_font("Arial", "B", 14)
            pdf.cell(0, 10, "Top Trigrams - Negative", ln=True)
            pdf.image(neg_filename, w=180)
            pdf.ln(10)

    # Simpan
    pdf.output("summary_report.pdf")
