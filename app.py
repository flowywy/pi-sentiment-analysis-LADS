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

clean_pycache()
clean_temp_folder()

from style import load_custom_style
from utils.model_utils import load_model, classify_sentiment
from utils.plot_utils import generate_wordcloud
from utils.preprocessing_utils import load_slang_dict, preprocess_dataframe
from streamlit_option_menu import option_menu

st.markdown(load_custom_style(), unsafe_allow_html=True)
st.markdown("<div class='title'>Sentiment Analysis of Love and Deepspace 💫</div>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("📌 Please upload a CSV/Excel file with a content column", type=['csv', 'xlsx'])
model, vectorizer = load_model()
slang_dict = load_slang_dict()

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
        st.error("❌ Column 'content' not found in the uploaded file.")
    else:
        with st.spinner("Analyzing sentiments, please wait..."):
            df = preprocess_dataframe(df, slang_dict)
            texts_to_classify = df['preprocess'].astype(str).tolist()
            df['Sentiment'] = classify_sentiment(model, vectorizer, texts_to_classify)

        st.success("✅ Sentiment classification completed")

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

        if selected == "Dataframe":
            st.subheader(f"Sentiment Classification Results (Total: {len(df)} rows)")
            st.dataframe(df[['content', 'Sentiment', 'preprocess']])

        elif selected == "Wordcloud":
            st.subheader("Wordcloud")
            positive_texts = df[df['Sentiment'] == 'positive']['content']
            if not positive_texts.empty:
                generate_wordcloud(positive_texts, "Positive")
            else:
                st.info("ℹ️ No Positive data found.")

            negative_texts = df[df['Sentiment'] == 'negative']['content']
            if not negative_texts.empty:
                generate_wordcloud(negative_texts, "Negative")
            else:
                st.info("ℹ️ No Negative data found.")

        elif selected == "Summary":
            st.subheader("Sentiment Summary")

        elif selected == "Download":
            st.subheader("⬇️ Download Results")
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Download as CSV", csv, "sentiment_results.csv", "text/csv")

st.markdown("---")
st.markdown("""
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
    """, unsafe_allow_html=True)
