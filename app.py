from flask import Flask, render_template, request, flash, redirect, url_for, send_file
import joblib
import logging
import pandas as pd
from io import BytesIO
from datetime import datetime
import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import nltk
# for text preprocessing
from nltk.stem import SnowballStemmer  # for stemming
from nltk.corpus import stopwords  # for getting stop words
import re  # for regular expression


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create a file handler and set the log file name
file_handler = logging.FileHandler('app.log')
file_handler.setLevel(logging.INFO)

# Create a stream handler to log to the console
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)

# Create a formatter and set it for both handlers
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(filename)s - line %(lineno)d - %(message)s")
file_handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)

# Add the handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(stream_handler)


# load models
logger.info("Loading model...")
translation_tokenizer = AutoTokenizer.from_pretrained("models/translation-tokenizer")
translation_model = AutoModelForSeq2SeqLM.from_pretrained("models/translation-model")
classifier_model = joblib.load('models/svm_model.pkl')
tfidf_vect = joblib.load('models/vectorizer.pkl')
nltk.download('stopwords')
logger.info("Loaded model")


def preprocess(text):
    # case folding (converting the string to lower case)
    text = text.lower()

    # removing html tags
    obj = re.compile(r"<.*?>")
    text = obj.sub(r" ", text)

    # removing url
    obj = re.compile(r"https://\S+|http://\S+")
    text = obj.sub(r" ", text)

    # removing punctuations
    obj = re.compile(r"[^\w\s]")
    text = obj.sub(r" ", text)

    # removing multiple spaces
    obj = re.compile(r"\s{2,}")
    text = obj.sub(r" ", text)

    # loading english stop words
    en_stopwords = stopwords.words('english')

    # removing stop words and stemming
    stemmer = SnowballStemmer("english")
    words = []

    text = [stemmer.stem(word) for word in text.split() if word not in en_stopwords]
    return " ".join(text)


def vectorize(preprocessed_text):
    x = tfidf_vect.transform([preprocessed_text])
    return x


def translate_german_to_english(text):
    # Tokenize the input text using the T5 tokenizer
    input_ids = translation_tokenizer.encode(text, return_tensors="pt")

    # Generate the translation using the T5 model
    translation = translation_model.generate(input_ids, max_length=1000)  # Adjust the max_length as needed

    # Decode the generated translation back into text
    translated_text = translation_tokenizer.decode(translation[0], skip_special_tokens=True)
    return translated_text


def predict_job(text):
    logger.info("Input Text: {}".format(text))
    translated_text = translate_german_to_english(text)
    logger.info("Translated Text: {}".format(translated_text))
    preprocessed_text = preprocess(translated_text)
    x = vectorize(preprocessed_text)
    return classifier_model.predict(x)[0]


app = Flask(__name__)
# Set the secret key
app.secret_key = os.urandom(24)

# Home page route
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        text = request.form['text']
        prediction = predict_job(text)
        logger.info("Prediction: {}".format(prediction))
        return render_template('home.html', prediction=prediction, text=text)
    else:
        text = """Ich würde gerne mehr über dieses Produkt erfahren:"""
        prediction = 'Product_Enquiries'
        return render_template('home.html', text=text, prediction=prediction)


@app.route('/bulk-classification', methods=['GET'])
def bulk_classification_page():
    return render_template('bulk-classification.html')


@app.route('/bulk-predict', methods=['POST'])
def bulk_predict():
    # Check if a file was uploaded
    if 'file' not in request.files:
        flash('No file uploaded', "danger")
        return redirect(url_for("bulk_classification_page"))

    file = request.files['file']
    inputDf = pd.read_csv(file, lineterminator='\n', delimiter=None, names=['text'])
    if inputDf.shape[0] > 50:
        flash("Too many input lines. Please upload file with max 50 Customer Queries with one query per line.",
              "danger")
        return redirect(url_for("bulk_classification_page"))
    logger.info("Bulk Classification Job Started")
    logger.info("Input Sentences: {}".format(inputDf.shape[0]))
    inputDf['prediction'] = inputDf['text'].apply(predict_job)

    # Generate the Excel file in memory
    excel_file = BytesIO()
    inputDf.to_excel(excel_file, index=False)
    excel_file.seek(0)  # Move the file pointer to the beginning of the file

    # Return the Excel file as a downloadable attachment
    timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    filename = f'{file.filename}-{timestamp}.xlsx'
    return send_file(excel_file, download_name=filename, as_attachment=True)


@app.route('/about')
def about():
    return render_template('about.html')


if __name__ == '__main__':
    app.run(debug=True, port=6050, use_reloader=False)
