from flask import Flask, render_template, request
import fitz
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

def read_text_from_pdf(uploaded_file):
    if uploaded_file.mimetype == "application/pdf":
        pdf_data = uploaded_file.read()
        with fitz.open(stream=pdf_data, filetype="pdf") as pdf_document:
            text = ''
            for page_num in range(pdf_document.page_count):
                page = pdf_document[page_num]
                text += page.get_text()
            return text
    else:
        return None

def calculate_similarity(resume_text, job_description_text):
    text = [resume_text, job_description_text]
    cv = CountVectorizer()
    count_matrix = cv.fit_transform(text)
    similarity = cosine_similarity(count_matrix)[0][1]
    match_percentage = round(similarity * 100, 2)
    return match_percentage

def match_best_job_description(resume_text, job_description_text):
    match_percentage = calculate_similarity(resume_text, job_description_text)
    return match_percentage

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        resume_file = request.files['resume']
        job_description_file = request.files['job_description']

        resume_text = read_text_from_pdf(resume_file)
        job_description_text = read_text_from_pdf(job_description_file)

        if resume_text is not None and job_description_text is not None:
            match_percentage = match_best_job_description(resume_text, job_description_text)
            return render_template('result.html', match_percentage=match_percentage)

    return render_template('Homepage.html')

@app.route('/u')
def u():
    return render_template('u.html')

@app.route('/feedback')
def feedback():
    return render_template('feedback.html')

@app.route('/recommendation')
def recommendation():
    return render_template('rec.html')

@app.route('/registration')
def registration():
    return render_template('registration.html')

@app.route('/log')
def log():
    return render_template('login.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/reset')
def reset():
    return render_template('reset.html')


@app.route('/sere')
def sere():
    return render_template('services.html')


if __name__ == '__main__':
    app.run(debug=True)
