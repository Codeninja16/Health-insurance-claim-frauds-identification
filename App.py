from flask import Flask, render_template, request, send_file
from sklearn.tree import DecisionTreeClassifier
from joblib import load, dump
import numpy as np
import pandas as pd
import os

app = Flask(_name_)

# Set the upload and output folders
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'output'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# Create directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load models and transformers
ss = load('standardscaler.joblib')
one_hot = load('one_hot_encoder.joblib')
pca = load('pca.joblib')
dt = load('decision_tree_model.joblib')  # Load the Decision Tree model

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'csv_file' not in request.files:
        return "No file found"

    csv_file = request.files['csv_file']
    
    # Validate file extension
    if not csv_file.filename.endswith('.csv'):
        return "Invalid file type. Only CSV files are allowed."

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], csv_file.filename)
    csv_file.save(file_path)

    try:
        raw_df = pd.read_csv(file_path)

        # Preprocess the DataFrame
        input_df = raw_df.copy()

        input_df['Hospital Id'] = input_df['Hospital Id'].astype(np.int64)
        input_df['ccs_diagnosis_code'] = input_df['ccs_diagnosis_code'].astype(np.int64)
        input_df['ccs_procedure_code'] = input_df['ccs_procedure_code'].astype(np.int64)
        input_df['Code_illness'] = input_df['Code_illness'].astype(np.int64)
        input_df['Days_spend_hsptl'] = input_df['Days_spend_hsptl'].astype(np.float64)
        input_df['baby'] = input_df['baby'].astype(np.int64)

        mask_numeric = input_df.dtypes == float
        numeric_cols = input_df.columns[mask_numeric]
        input_df[numeric_cols] = ss.transform(input_df[numeric_cols])

        mask = input_df.dtypes == object
        object_cols = input_df.columns[mask]
        input_df = one_hot.transform(input_df)
        names = one_hot.get_feature_names_out()
        column_names = [name[name.find("") + 1:] for name in [name[name.find("_") + 2:] for name in names]]
        input_df = input_df.toarray()
        input_df = pd.DataFrame(data=input_df, columns=column_names)
        input_df_hat = pca.transform(input_df)
        input_df_hat_PCA = pd.DataFrame(columns=[f'Projection on Component {i + 1}' for i in range(len(input_df.columns))], data=input_df_hat)
        input_df_hat_PCA = input_df_hat_PCA.iloc[:, :4]

        # Predict
        predictions = dt.predict(input_df_hat_PCA)  # Use the Decision Tree model
        raw_df['predicted_class'] = predictions
        raw_df['predicted_class'] = raw_df['predicted_class'].apply(lambda x: 'Fraud' if x == 1 else 'Not Fraud')
        
        # Save output
        output_file = os.path.join(app.config['OUTPUT_FOLDER'], 'output.csv')
        raw_df.to_csv(output_file, index=False)

    except Exception as e:
        return f"An error occurred: {str(e)}"

    return render_template('result.html', input_file=csv_file.filename, output_file='output.csv')

@app.route('/download_output')
def download_output():
    output_file = os.path.join(app.config['OUTPUT_FOLDER'], 'output.csv')
    if os.path.exists(output_file):
        return send_file(output_file, as_attachment=True)
    return "Output file not found."

@app.route('/download_text')
def download_text():
    text_file = 'text_file.txt'
    if os.path.exists(text_file):
        return send_file(text_file, as_attachment=True)
    return "Text file not found."

if _name_ == '_main_':
    app.run(debug=True)

 
