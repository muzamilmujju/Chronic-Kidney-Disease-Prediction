from flask import Flask, render_template, request
import pickle
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

app = Flask(__name__)

def predict(values, dic):
    if len(values) == 8:
        model = pickle.load(open('models/diabetes.pkl','rb'))
        values = np.asarray(values)
        return model.predict(values.reshape(1, -1))[0]
    elif len(values) == 26:
        model = pickle.load(open('models/breast_cancer.pkl','rb'))
        values = np.asarray(values)
        return model.predict(values.reshape(1, -1))[0]
    elif len(values) == 13:
        model = pickle.load(open('models/heart.pkl','rb'))
        values = np.asarray(values)
        return model.predict(values.reshape(1, -1))[0]
    elif len(values) == 18:
        model = pickle.load(open('models/kidney.pkl','rb'))
        values = np.asarray(values)
        return model.predict(values.reshape(1, -1))[0]
    elif len(values) == 10:
        model = pickle.load(open('models/liver.pkl','rb'))
        values = np.asarray(values)
        return model.predict(values.reshape(1, -1))[0]

@app.route("/")
def home():
    return render_template('home.html')

@app.route("/diabetes", methods=['GET', 'POST'])
def diabetesPage():
    return render_template('diabetes.html')

@app.route("/cancer", methods=['GET', 'POST'])
def cancerPage():
    return render_template('breast_cancer.html')

@app.route("/heart", methods=['GET', 'POST'])
def heartPage():
    return render_template('heart.html')

@app.route("/kidney", methods=['GET', 'POST'])
def kidneyPage():
    return render_template('kidney.html')

@app.route("/liver", methods=['GET', 'POST'])
def liverPage():
    return render_template('liver.html')

@app.route("/malaria", methods=['GET', 'POST'])
def malariaPage():
    return render_template('malaria.html')

@app.route("/pneumonia", methods=['GET', 'POST'])
def pneumoniaPage():
    return render_template('pneumonia.html')

@app.route("/predict", methods=['GET', 'POST'])
def predictPage():
    if request.method == 'POST':
        try:
            # Convert form data to float list
            to_predict_dict = request.form.to_dict()
            to_predict_list = list(map(float, to_predict_dict.values()))

            # Predict using the appropriate model
            pred = predict(to_predict_list, to_predict_dict)

            # Debug print to terminal
            print("Model Prediction Output:", pred)

            # Handle prediction output for kidney disease (adjust as needed)
            if str(pred).lower() in ['1', 'true', 'yes', 'present']:
                result = "Kidney Disease Detected"
            else:
                result = "No Kidney Disease Detected"

            # Render kidney.html with prediction result
            return render_template('kidney.html', prediction=result)

        except Exception as e:
            # Show error message on the kidney page
            message = "Please enter valid data. Error: " + str(e)
            return render_template('kidney.html', message=message)

    # GET request just shows the form
    return render_template('kidney.html')

@app.route("/malariapredict", methods=['POST', 'GET'])
def malariapredictPage():
    if request.method == 'POST':
        try:
            if 'image' in request.files:
                img = Image.open(request.files['image'])
                img = img.resize((36,36))
                img = np.asarray(img)
                img = img.reshape((1,36,36,3))
                img = img.astype(np.float64)
                model = load_model("models/malaria.h5")
                pred = np.argmax(model.predict(img)[0])
                return render_template('malaria_predict.html', pred=pred)
        except:
            message = "Please upload an Image"
            return render_template('malaria.html', message=message)
    return render_template('malaria.html')

@app.route("/pneumoniapredict", methods=['POST', 'GET'])
def pneumoniapredictPage():
    if request.method == 'POST':
        try:
            if 'image' in request.files:
                img = Image.open(request.files['image']).convert('L')
                img = img.resize((36,36))
                img = np.asarray(img)
                img = img.reshape((1,36,36,1))
                img = img / 255.0
                model = load_model("models/pneumonia.h5")
                pred = np.argmax(model.predict(img)[0])
                return render_template('pneumonia_predict.html', pred=pred)
        except:
            message = "Please upload an Image"
            return render_template('pneumonia.html', message=message)
    return render_template('pneumonia.html')

if __name__ == '__main__':
    app.run(debug=True)
