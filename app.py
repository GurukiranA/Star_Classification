from flask import Flask, render_template, request
import jsonify
import requests
import pickle
import numpy as np
import sklearn
import joblib
from sklearn.preprocessing import StandardScaler
app = Flask(__name__)
model_tree = joblib.load(open('star_model_mass.pkl', 'rb'))
sc=joblib.load(open('minmax1.pkl','rb'))
one_hot=joblib.load(open('onehot1.pkl','rb'))
@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')


standard_to = StandardScaler()
@app.route("/predict", methods=['POST'])
def predict():

    if request.method == 'POST':
        Temperature = int(request.form['Temperature'])
        Relative_Luminosity=float(request.form['Relative_Luminosity'])
        Radius=float(request.form['Radius'])
        Absolute_Magnitude=float(request.form['Absolute_Magnitude'])
        Mass_of_Star = float(request.form['Mass_of_Star'])
        
        Color=request.form['Color']
        
        
        Spectral_Class=request.form["Spectral_Class"]
        
        num_values = sc.transform([[Temperature, Relative_Luminosity, Radius, Absolute_Magnitude,Mass_of_Star]])
        cat_values = one_hot.transform(np.array([Color, Spectral_Class]).reshape(-1, 2))
        
        array = []
        for i in num_values:
            for j in i:
                array.append(j)
        for i in cat_values:
            for j in i:
                array.append(j)
        
        
    
        prediction = model_tree.predict([array])
        if prediction[0]==[0]:
            return render_template('index.html',prediction_text="the star is Red Dwarf")
        elif prediction[0]==[1]:
            return render_template('index.html',prediction_text="the star is Brown Dwarf")
        elif prediction[0]==[2]:
            return render_template('index.html',prediction_text="the star is Main Sequence ")
        elif prediction[0]==[3]:
            return render_template('index.html',prediction_text="the star is Supergiant")
        elif prediction[0]==[4]:
            return render_template('index.html',prediction_text="the star is Hypergiant")
        else :
            return render_template('index.html',prediction_text="hmmm")
            
if __name__=="__main__":
    app.run(debug=True)