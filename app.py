from flask import Flask, render_template, url_for, request, redirect

import requests

import os


from keras.models import load_model
from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences

app = Flask(__name__, template_folder='Template')


model = load_model ('CNN_model.hdf5')

"""
Routes
"""

@app.route('/', methods=['GET','POST'])
def index():
    if request.method == 'POST':
        try:
            input_review = str(request.form.get("review"))
        except requests.exceptions.ConnectionError:
            print("Connection refused")



        word_to_index = imdb.get_word_index()
        #print("Processing input...")
        tokenized_review = [word.lower() for word in input_review.split()]
        text_indices = [word_to_index[word]+3 for word in tokenized_review if word in word_to_index.keys()]
        model_input = [text_indices]
        model_input = pad_sequences (model_input, maxlen = 2494)

        #print("Running the model with input...")
        prediction = model.predict(model_input, verbose = 0)[0][0]


        #print(prediction)
        sentiment = ""
        description = "0 denotes most negative ; 1 denotes most positive"

        """result = float(result_json['outputs']['prediction'][0][0])
        description = result_json['outputs']['description']
        sentiment = ""
        print(description)
        print(result)"""

        if prediction < 0.5:
            sentiment += 'Negative'
    
        else:
            sentiment += 'Positive'
            
        if prediction != None:
            return render_template('result.html', result = prediction, review = input_review, description= description, sentiment= sentiment)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug = True)
