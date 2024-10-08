# New York Housing Market Prediction and Deployment

## 1. Dataset Exploration and Preprocessing
- Load and explore the dataset.
- Handle missing values and remove duplicates.
- Feature engineering: create new features such as LOG_PRICE (log-transformed PRICE) and clean the LOCALITY and SUBLOCALITY columns.
- Deal with outliers using methods like IQR or Z-scores.
- Use Sweetviz and Ydata Profiling for exploratory data analysis (EDA) and automated data reporting.

## 2. Model Training
- Define features (X) and target variable (y).
- Split the data into training and testing sets.
- Preprocessing with StandardScaler and OneHotEncoder.
- Train multiple models: XGBRegressor, RandomForestRegressor, and GradientBoostingRegressor.
- Perform hyperparameter tuning using GridSearchCV to improve model performance.

## 3. Evaluation
- Evaluate models using mean_squared_error, mean_absolute_error, and R^2 score.
- Based on your metrics, XGBRegressor performed best with an R^2 score of 0.7749.
- Visualize the results using scatter plots for actual vs predicted values.

## 4. Model Saving
- Save the best XGBRegressor model using pickle.

## 5. Flask API Deployment
- Set up a simple Flask web application for real-time predictions.
- Define the input parameters such as property_type, neighborhood, bedrooms, baths, and property_sqft.
- Load the saved model (xgb_model.pkl) and make predictions in the Flask app.
- Un-log the predicted house price and return it in a user-friendly format.

## 6. UI Integration (HTML)
- Use templates/index.html to create a user interface where users can input property details (Type, Beds, Baths, etc.) and get predictions.
- Deploy the app on localhost or any cloud service (Heroku, AWS) for public use.

## 7. Example Input for Prediction
- Example input:
{
  "property_type": "Condo for sale",
  "neighborhood": "Manhattan",
  "bedrooms": 3,
  "baths": 2,
  "property_sqft": 1500
}
- Example output:
{
  "prediction_text": "House price should be $1,234,567.89"
}

## 8. Deployment Architecture
- Frontend: HTML forms to take input from users.
- Backend: Flask API serves predictions based on the trained XGBRegressor model.
- Static Files: Store assets like CSS or images in the static folder.
