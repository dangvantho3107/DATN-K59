from flask import Flask, render_template, request
import pickle
import regex as re

app = Flask (__name__)

# load the model from disk
filename = 'models\clf_SVM.sav'
clf = pickle.load(open(filename, 'rb'))

# load the vectorizer from disk
filename = 'models\_vectorizer.sav'
vectorizer = pickle.load(open(filename, 'rb'))

def preprocess_input(mail):
    mail = re.sub(r"[^a-zA-Z0-9]+", ' ', mail)
    demo = vectorizer.transform([mail]).toarray()
    return demo

# xu ly khi ban vao host 127.0.0.1:5000
@app.route('/')
def index():
    return render_template('main.html')  # se hien ra giao dien la file main.html



@app.route('/test.html')
def test():
    return render_template('test.html')  # se hien ra giao dien la file test.html

@app.route('/result', methods=['POST'])
def predict():

    if request.method == 'POST':
        message = request.form['message']
        demo_email = preprocess_input(message)
        my_prediction = clf.predict(demo_email)

    return render_template('result.html', prediction=my_prediction)

if __name__ == "__main__":
    app.run()