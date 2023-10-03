import pickle
from sklearn.datasets import load_iris
from flask import Flask, request, render_template

iris = load_iris()
app = Flask(__name__)

# Load the trained model from the pickle file
with open('iris_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/', methods=['GET', 'POST'])
def predict_iris():
    if request.method == 'POST':
        try:
            sepal_length = float(request.form['sepal_length'])
            sepal_width = float(request.form['sepal_width'])
            petal_length = float(request.form['petal_length'])
            petal_width = float(request.form['petal_width'])

            input_features = [[sepal_length, sepal_width, petal_length, petal_width]]
            predicted_class = model.predict(input_features)[0]
            predicted_species = iris.target_names[predicted_class]

            return render_template('index.html', prediction=f"The predicted species is: {predicted_species}")
        except ValueError:
            return render_template('index.html', prediction="Invalid input. Please enter numeric values.")

    return render_template('index.html', prediction="")

if __name__ == '__main__':
    app.run(debug=True)
