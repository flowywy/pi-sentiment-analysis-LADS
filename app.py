import streamlit as st
import pandas as pd
import io
import shutil
import os

# Tambahkan fungsi clean
def clean_pycache():
    if os.path.exists("__pycache__"):
        shutil.rmtree("__pycache__")

def clean_temp_folder():
    if os.path.exists("temp"):
        shutil.rmtree("temp")

# Bersihkan cache
clean_pycache()
clean_temp_folder()

# Import custom style
from style import load_custom_style

# Import utils
from utils.model_utils import load_model, classify_sentiment
from utils.plot_utils import (
    generate_wordcloud, save_wordcloud_image,
    get_top_ngrams, plot_top_ngrams_bar_chart,
    save_summary_chart
)
from utils.pdf_utils import create_summary_pdf
from utils.preprocessing_utils import load_slang_dict, preprocess_dataframe
from streamlit_option_menu import option_menu

# CUSTOM STYLE
st.markdown(load_custom_style(), unsafe_allow_html=True)

# MAIN TITLE
st.markdown("<div class='title'>Sentiment Analysis of Love and Deepspace 💫</div>", unsafe_allow_html=True)

# NOTE
st.markdown("<p class='note-text'>📌 Please upload a CSV/Excel file with a content column</p>", unsafe_allow_html=True)

# FILE UPLOADER
uploaded_file = st.file_uploader("", type=['csv', 'xlsx'])

# LOAD MODEL & VECTORIZER
model, vectorizer = load_model()

# LOAD SLANG DICTIONARY
slang_dict = load_slang_dict()

# MAIN PROGRAM
if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"⚠️ Failed to read the file: {e}")
        st.stop()

    if 'content' not in df.columns:
        st.error("❌ Column 'content' not found in the uploaded file.", unsafe_allow_html=True)
    else:
        with st.spinner("Analyzing sentiments, please wait..."):
            df = preprocess_dataframe(df, slang_dict)
            texts_to_classify = df['preprocess'].astype(str).tolist()
            df['Sentiment'] = classify_sentiment(model, vectorizer, texts_to_classify)

        st.success("✅ Sentiment classification completed")

        # SIDEBAR MENU (Muncul setelah file di-upload)
        with st.sidebar:
            selected = option_menu(
                menu_title="Menu",
                options=["Dataframe", "Wordcloud", "Summary", "Download"],
                icons=["table", "cloud", "bar-chart", "download"],
                menu_icon="cast",
                default_index=0,
                styles={
                    "container": {"background-color": "#725CAD"},
                    "icon": {"color": "#2A1458", "font-size": "20px"},
                    "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0px", "--hover-color": "#d6c8ff"},
                    "nav-link-selected": {"background-color": "#EBD6FB", "color": "#2A1458"},
                }
            )

        # MENU: Dataframe
        if selected == "Dataframe":
            st.subheader(f"Sentiment Classification Results (Total: {len(df)} rows)")
            st.dataframe(df[['content', 'Sentiment', 'preprocess']])

        # MENU: Wordcloud
        elif selected == "Wordcloud":
            st.subheader("Wordcloud")
            
            positive_texts = df[df['Sentiment'] == 'positive']['content']
            negative_texts = df[df['Sentiment'] == 'negative']['content']

            col1, col2 = st.columns(2)

            with col1:
                if not positive_texts.empty:
                    generate_wordcloud(positive_texts, "Positive")
                else:
                    st.info("ℹ️ No Positive data found.")

            with col2:
                if not negative_texts.empty:
                    generate_wordcloud(negative_texts, "Negative")
                else:
                    st.info("ℹ️ No Negative data found.")

        # MENU: Summary
        elif selected == "Summary":
            st.subheader("Sentiment Summary")

            count_positive = (df['Sentiment'] == 'positive').sum()
            count_negative = (df['Sentiment'] == 'negative').sum()
            total = count_positive + count_negative

            percent_positive = round(100 * count_positive / total, 1) if total > 0 else 0
            percent_negative = round(100 * count_negative / total, 1) if total > 0 else 0

            summary_df = pd.DataFrame({
                'Sentiment': ['Positive', 'Negative'],
                'Count': [count_positive, count_negative]
            })
            st.bar_chart(summary_df.set_index('Sentiment'))

            if count_positive > count_negative:
                summary_text = f"<span style='color:green; font-weight:bold;'>😊 This Data Contains More positives. ({percent_positive}% positive)</span>"
            elif count_negative > count_positive:
                summary_text = f"<span style='color:red; font-weight:bold;'>☹️ This Data Contains More negatives. ({percent_negative}% negative)</span>"
            else:
                summary_text = f"<span style='color:gray; font-weight:bold;'>⚖️ Equal number of positives and negatives.</span>"
            st.markdown(f"**Summary:** {summary_text}", unsafe_allow_html=True)

            st.markdown("#### Average Text Length per Sentiment")
            df['TextLength'] = df['content'].astype(str).apply(lambda x: len(x.split()))
            avg_length_positive = df[df['Sentiment'] == 'positive']['TextLength'].mean()
            avg_length_negative = df[df['Sentiment'] == 'negative']['TextLength'].mean()

            st.write(f"**Positive:** {avg_length_positive:.1f} words on average")
            st.write(f"**Negative:** {avg_length_negative:.1f} words on average")

            st.markdown("#### Example Positive Texts")
            positive_examples = df[df['Sentiment'] == 'positive']['content'].head(3).tolist()
            for i, text in enumerate(positive_examples, 1):
                st.write(f"**{i}.** {text}")

            st.markdown("#### Example Negative Texts")
            negative_examples = df[df['Sentiment'] == 'negative']['content'].head(3).tolist()
            for i, text in enumerate(negative_examples, 1):
                st.write(f"**{i}.** {text}")

            ngram_range = (3, 3)
            positive_texts = df[df['Sentiment'] == 'positive']['content']
            if not positive_texts.empty:
                st.markdown("#### Top Trigrams in Positive Sentiments")
                top_positive_ngrams = get_top_ngrams(positive_texts, ngram_range=ngram_range)
                plot_top_ngrams_bar_chart(top_positive_ngrams, "Top Trigrams - Positive")

            negative_texts = df[df['Sentiment'] == 'negative']['content']
            if not negative_texts.empty:
                st.markdown("#### Top Trigrams in Negative Sentiments")
                top_negative_ngrams = get_top_ngrams(negative_texts, ngram_range=ngram_range)
                plot_top_ngrams_bar_chart(top_negative_ngrams, "Top Trigrams - Negative")

        # MENU: Download
        elif selected == "Download":
            st.subheader("⬇️ Download Results")

            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Download as CSV", csv, "sentiment_results.csv", "text/csv")

            buffer = io.BytesIO()
            df.to_excel(buffer, index=False)
            buffer.seek(0)
            st.download_button("⬇️ Download as Excel", buffer, "sentiment_results.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

            if st.button("📄 Create Summary PDF"):
                if not os.path.exists("temp"): os.makedirs("temp")
                positive_texts = df[df['Sentiment'] == 'positive']['content']
                negative_texts = df[df['Sentiment'] == 'negative']['content']

                count_positive = (df['Sentiment'] == 'positive').sum()
                count_negative = (df['Sentiment'] == 'negative').sum()
                total = count_positive + count_negative
                percent_positive = round(100 * count_positive / total, 1) if total > 0 else 0
                percent_negative = round(100 * count_negative / total, 1) if total > 0 else 0

                if not positive_texts.empty:
                    save_wordcloud_image(positive_texts, "temp/wordcloud_positive.png")
                if not negative_texts.empty:
                    save_wordcloud_image(negative_texts, "temp/wordcloud_negative.png")
                save_summary_chart(df, "temp/summary_chart.png")

                create_summary_pdf(
                    df,
                    count_positive, count_negative,
                    percent_positive, percent_negative,
                    not positive_texts.empty, not negative_texts.empty
                )

                st.success("Summary created. Please click button below to download.")
                with open("summary_report.pdf", "rb") as f:
                    st.download_button("📄 Download Summary PDF", f, file_name="summary_report.pdf", mime="application/pdf")

                clean_temp_folder()
                clean_pycache()

# FOOTER / ABOUT APP
st.markdown("---")
st.markdown(
    """
    <div style='
        text-align: center; 
        color: white; 
        font-size: 14px;
        background-color: #725CAD; 
        padding: 10px; 
        border-radius: 8px;
        '>
        <b>About this App</b><br>
        This sentiment analysis app was built using <b>Streamlit</b> and a <b>Naive Bayes</b> model with accuracy 87%<br>
        Designed to analyze reviews of <i>Love and Deepspace</i> game.<br>
    </div>
    """,
    unsafe_allow_html=True
)
