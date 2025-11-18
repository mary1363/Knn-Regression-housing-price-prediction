# Knn-Regression-housing-price-prediction
Machine-learning model for predicting house prices and price-per-square-foot using linear regression, KNN-based region assignment, feature interactions, and scaled training data. Includes full pipeline for preprocessing, modeling, evaluation, and real-world prediction.

# House Price per SqFt Prediction

## Overview
This project predicts house prices and price per square foot using linear regression and KNN region modeling.

## Dataset
- 5,000 houses
- Features: rooms, garage, sqft, kitchen features, floor covering, year built, latitude/longitude, etc.

## Modeling
- KNN to assign region based on latitude/longitude + price category
- Multi-variable linear regression with feature interactions
- Scaled features for gradient descent

## Usage
```python
from model import predict_house_price, get_price_category

sample_house = {
    'rooms': 4, 'garage': 2, 'year_built': 2005,
    'sqft': 2400, 'kitchen_feat': 3, 'fireplaces': 1,
    'floor_cover': 2, 'latitude': 32.22, 'longitude': -110.78
}

region = get_price_category(sample_house['latitude'], sample_house['longitude'], knn_model)
sample_house['region_knn'] = region

pred_price, price_per_sqft = predict_house_price(sample_house, model, mu_int, sigma_int)
print(pred_price, price_per_sqft)
