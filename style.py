def load_custom_style():
    return """
    <style>
    .stApp {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .title {
        font-size: 42px;
        font-weight: bold;
        color: #7F55B1;
        text-align: center;
        margin-bottom: 20px;
    }

    .stFileUploader > div {
        background-color: #725CAD;  
        border-radius: 12px;
        padding: 15px;
        border: 2px dashed #B74E91;
        margin-top: 5px;  
        margin-bottom: 10px; 
    }
    .stFileUploader > div > label > div {
        background-color: #B74E91 !important;
        color: white !important;
        border-radius: 10px !important;
        padding: 8px 15px !important;
        font-weight: bold !important;
        cursor: pointer;
        text-align: center;
    }
    </style>
    """
