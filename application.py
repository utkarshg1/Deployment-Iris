import pandas as pd
import pickle 
from flask import Flask, render_template, request

# Intializae flask app
application = Flask(__name__)
app = application

@app.route('/')
def home_page():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_species():
    if request.method=='GET':
        return render_template('index.html')
    else:
        # Load preprocessor and model
        with open('notebook/preprocessor.pkl', 'rb') as file1:
            pre = pickle.load(file1)
        with open('notebook/IrisModel.pkl', 'rb') as file2:
            model = pickle.load(file2)
        # Get the sep_len, sep_wid
        sep_len = float(request.form.get('sepal_length'))
        sep_wid = float(request.form.get('sepal_width'))
        pet_len = float(request.form.get('petal_length'))
        pet_wid = float(request.form.get('petal_width'))
        # Convert above into dataframe
        xnew = pd.DataFrame([sep_len, sep_wid, pet_len, pet_wid]).T
        cols = pre.get_feature_names_out()
        xnew.columns = cols
        # Preprocessing
        xnew_pre = pre.transform(xnew)
        xnew_pre = pd.DataFrame(xnew_pre, columns=cols)
        # Prediction
        prediction = model.predict(xnew_pre)[0]
        # Probability
        prob = model.predict_proba(xnew_pre).max()
        prob = round(prob, 4)
        return render_template('index.html', prediction=prediction, prob=prob)        

if __name__=='__main__':
    app.run(host='0.0.0.0', debug=False)