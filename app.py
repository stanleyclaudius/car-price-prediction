from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)
df = pd.read_csv('cleaned_car_price.csv')
lr_model = pickle.load(open('CarPricePredictorModel.pkl', 'rb'))

@app.route('/')
def index():
    manufacturers = sorted(df['company'].unique())
    models = sorted(df['name'].unique())
    years = sorted(df['year'].unique(), reverse=True)
    fuels = sorted(df['fuel_type'].unique())
    return render_template('index.html', manufacturers=manufacturers, models=models, years=years, fuels=fuels)

@app.route('/predict', methods=['POST'])
def predict():
    manufacturer = request.form.get('manufacturer')
    model = request.form.get('model')
    year = request.form.get('year')
    fuel = request.form.get('fuel')
    km_driven = request.form.get('km_driven')

    prediction = lr_model.predict(pd.DataFrame([[model, manufacturer, year, km_driven, fuel]], columns=['name', 'company', 'year', 'kms_driven', 'fuel_type']))
    return str(np.round(prediction[0], 2))

if __name__ == '__main__':
    app.run(debug=True)