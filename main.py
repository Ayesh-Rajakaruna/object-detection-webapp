from flask import Flask, render_template, request
from Entity import prediction
from CreateModels import makemodel
import os

app = Flask(__name__)


@app.route("/")
@app.route("/home", methods=['GET', 'POST'])
def home():
    return render_template('Home.html')


@app.route("/predict", methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        file_name = file.filename
        file_path = os.path.join(r'.\static\temporary', "image.jpg")
        if file_name == "":
            massage = "Please choose the file"
            return render_template('error.html', massage=massage)
        elif file_name.split(".")[-1].upper() not in ["JPEG", "JPG", "PNG"]:
            massage = "Please choose the valid image file"
            return render_template('error.html', massage=massage)
        else:
            file.save(file_path)
            result, accuracy = prediction(file_path)
            if result == "No model":
                return render_template('error.html', massage=result)
            else:
                return render_template('predict.html', product=result, user_image=file_path, Accuracy=accuracy)


@app.route("/makeModel", methods=['GET', 'POST'])
def make_model():
    if request.method == 'POST':
        iteration = request.form.get("iteration")
        massage = makemodel("DataSet", iteration)
        if massage == "successfully model created":
            return render_template('Finalmodelc.html', massage=massage)
        else:
            return render_template('Finalmodelw.html', massage=massage)
    else:
        return render_template('Loadmodel.html')


if __name__ == "__main__":
    app.run()
