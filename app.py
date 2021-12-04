# -*- coding: utf-8 -*-
from flask import Flask,render_template,request,session,redirect,url_for
#from module import prediction
import numpy as np
import pickle
#from sklearn.ensemble import RandomForestClassifier
app = Flask(__name__)
app.secret_key="wine"
modelm=pickle.load(open('model1.pkl','rb'))
@app.route('/home')
def home():
    return render_template('index.html')

@app.route('/model', methods=["POST", "GET"])
def model():
    if request.method=='POST':
        volatile_acidity=request.form['va']
        chlorides=request.form['cl']
        total_sulfur_dioxide=request.form['tsd']
        density=request.form['de']
        sulphates=request.form['su']
        alcohol=request.form['alc']
        

        session['volatile_acidity']=volatile_acidity
        session['chlorides']=chlorides
        session['total_sulfur_dioxide']=total_sulfur_dioxide
        session['density']=density
        session['sulphates']=sulphates
        session['alcohol']=alcohol  
        return redirect(url_for('predict_m'))
    else:  
        return render_template('model.html')

@app.route('/prediction')
def predict_m():
    input_data=[session['volatile_acidity']
            ,session['chlorides'],session['total_sulfur_dioxide'],
            session['density'],session['sulphates'],session['alcohol']]
    input_data = np.asarray(input_data)
    input_data_reshaped = input_data.reshape(1,-1)
    prediction=modelm.predict(input_data_reshaped)
    #predictions=session["prediction"]
    if prediction[0]==0:
        reverb='0'
    elif prediction[0]==1:
        reverb='1'
    elif prediction[0]==2:
        reverb='2'   
    elif prediction[0]==3:
        reverb='3'
    elif prediction[0]==4:
        reverb='4'    
    else:
        reverb='Thats some fine Quality wine'
    return render_template('prediction.html',predicts=reverb)

if __name__ == "__main__":
    app.run(debug=True)