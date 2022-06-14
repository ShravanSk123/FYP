from flask import Flask, render_template, request
import warnings
import pickle
import functions as func

from werkzeug import secure_filename


warnings.filterwarnings("ignore")
main_app = Flask(__name__)

# Application Home Page
@main_app.route("/")
def index():
    return render_template("index.html", page_title = "Text Summarization & Categorization")

# 1. Take input directly
@main_app.route("/inp_text", methods=['GET', 'POST'])
def inp_text():
    if request.method == 'POST':
        input_text = request.form['text_input_text']
        classifier = request.form['text_classifier']
        sentences_number = request.form['text_sentences_number']
        classifier_model = pickle.load(open('ML_models/' + classifier + '.pkl', 'rb'))
        text_summary, text_category = func.TSC(input_text, sentences_number, classifier_model)
     
    return render_template("index.html", page_title="Text Summarization & Categorization", input_text=input_text, text_summary=text_summary, text_category=text_category)


# 2. Take input through URL
@main_app.route("/inp_url", methods=['GET', 'POST'])
def inp_url():
    if request.method == 'POST':
        input_url = request.form['url_input_text']
        input_text = func.fetch_data(input_url)
        classifier = request.form['url_classifier']
        sentences_number = request.form['url_sentences_number']
        classifier_model = pickle.load(open('ML_models/' + classifier + '.pkl', 'rb'))
        text_summary, text_category = func.TSC(input_text, sentences_number, classifier_model)
    return render_template("index.html", page_title="Text Summarization & Categorization", input_text=input_text, text_summary=text_summary, text_category=text_category)


# 3. Take input through file
@main_app.route("/inp_file", methods=['GET', 'POST'])
def inp_file():
    if request.method == 'POST':
        f = request.files['file']
        f.save(secure_filename(f.filename))

        saved_file = open(f.filename)
        input_text = saved_file.read()

        classifier = request.form['file_classifier']
        sentences_number = request.form['file_sentences_number']
        classifier_model = pickle.load(open('ML_models/' + classifier + '.pkl', 'rb'))
        text_summary, text_category = func.TSC(input_text, sentences_number, classifier_model)
    return render_template("index.html", page_title="Text Summarization & Categorization", input_text=input_text, text_summary=text_summary, text_category=text_category)

if __name__ == "__main__":
    from waitress import serve
    #serve(main_app, host="0.0.0.0", port=8080)
    main_app.run(debug=True)