# app.py
from flask import Flask, render_template, request, redirect, url_for, flash, send_file
import os
import pandas as pd
from werkzeug.utils import secure_filename
from functions.processing import clean_data_pipeline, check_dataset_quality, generate_confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import secrets

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)
app.config['UPLOAD_FOLDER'] = 'test'
app.config['PROCESSED_FOLDER'] = 'cleaned_data'

# Allowed file extensions
ALLOWED_EXTENSIONS = {'csv', 'xlsx'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Run data cleaning pipeline
        output_folder = app.config['PROCESSED_FOLDER']
        cleaned_df = clean_data_pipeline(file_path, output_folder)
        
        # Model quality check and Confusion matrix
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        quality_check = check_dataset_quality(cleaned_df, model=model)
        
        if quality_check:
            generate_confusion_matrix(cleaned_df, model=model)
        
        return render_template('results.html', quality_check=quality_check, filename=filename)
    else:
        flash('Allowed file types are csv, xlsx')
        return redirect(request.url)

if __name__ == '__main__':
    app.run(debug=True)
