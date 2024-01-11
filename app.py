from flask import Flask, render_template, request

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline

app = Flask(__name__)

pretrained = "mdhugol/indonesia-bert-sentiment-classification"
model = AutoModelForSequenceClassification.from_pretrained(pretrained)
tokenizer = AutoTokenizer.from_pretrained(pretrained)
sentiment_analysis = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

label_index = {'LABEL_0': 'positive', 'LABEL_1': 'neutral', 'LABEL_2': 'negative'}

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        text = request.form['text']
        result = sentiment_analysis(text)
        status = label_index[result[0]['label']]
        score = result[0]['score'] * 100
        return render_template('index.html', text=text, status=status, score=score)
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)