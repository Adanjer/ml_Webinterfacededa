from flask import Flask, render_template, request, jsonify
from joblib import load
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the models during app initialization
models = {
    'KNN': load("model/knn_model.pkl"),
    'Naive Bayes': load("model/naive_bayes_model.pkl"),
    'Decision Tree': load("model/decision_tree_model.pkl")
}

print("Models loaded successfully:", {name: type(model) for name, model in models.items()})

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/classify", methods=['POST'])
def classify_data():
    try:
        # Retrieve the classifier selected by the user
        chosen_model = request.form.get('classifier')
        print(f"Classifier selected: {chosen_model}")

        # Extract and parse input features from the form
        feature_names = [
            'Clump_thickness', 'Uniformity_Cell_Size', 'Uniformity_Cell_Shape', 
            'Marginal_Adhesion', 'Single_Epithelial_Cell_Size', 
            'Bland_Chromatin', 'Normal_Nucleoli', 'Mitoses'
        ]
        
        # Convert the form data into a numerical list
        input_features = [float(request.form.get(feature, 0)) for feature in feature_names]
        input_df = pd.DataFrame([input_features], columns=feature_names)

        # Check the chosen model and predict accordingly
        if chosen_model not in models:
            chosen_model = 'KNN'  # Default to KNN if no valid classifier is chosen
        prediction = models[chosen_model].predict(input_df)

        # Interpret the prediction result
        result = "Benign" if prediction[0] == 2 else "Malignant" if prediction[0] == 4 else "Unknown"

        # Format the response with the inputs and prediction
        response = {
            'prediction': result,
            'classifier': chosen_model,
            'user_input': dict(zip(feature_names, input_features))
        }

        return jsonify(response)

    except Exception as error:
        print(f"Error occurred: {error}")
        return jsonify({'error': str(error)}), 400

if __name__ == "__main__":
    app.run(debug=True)
