from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__, template_folder = "template")
model = pickle.load(open("Linear_model.pkl",'rb'))
scaler = pickle.load(open("scaler.pkl", 'rb')) # this is for using scaler 

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods =["POST"] )
def pred():
    #fetch the data from ui

    age = int(request.form.get("age"))
    bmi = float(request.form.get("bmi"))
    children = int(request.form.get("children"))
    sex = float(request.form.get("sex"))
    smoker = float(request.form.get("smoker"))

    region = request.form.get("region")
    r_nw, r_se, r_sw = 0, 0, 0
    if region == "northwest": 
        r_nw = 1
    elif region == "southeast":
        r_se = 1
    elif region == "southwest": 
        r_sw = 1    

    features = [[age, sex, bmi, children, smoker, r_nw, r_se, r_sw]]

    #scale the feature, as the trained dataset scaled
    scaled_features = scaler.transform(features)
    log_prediction = model.predict(scaled_features)
    final_predict = np.expm1(log_prediction[0])




    return render_template("index.html", prediction_text = f"Estimated insurance price is {final_predict}")


if __name__ == "__main__":
    app.run(debug= True)