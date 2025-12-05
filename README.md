# 🏠 Bengaluru House Price Prediction (Machine Learning + Streamlit App)

This project predicts **house prices in Bengaluru** using Machine Learning.  
It includes complete data cleaning, feature engineering, outlier removal, model training, evaluation, saving the model, and a **Streamlit web app** for real-time predictions.

This project uses the **Bengaluru House Price Dataset (Kaggle)** and implements the model **without Scikit-Learn Pipelines** to make preprocessing steps more transparent and easier to learn.

---

## 🚀 Project Features

- Predicts house prices in **Lakhs (₹)**  
- Cleans and prepares raw Bengaluru housing data  
- Extracts BHK from textual data  
- Cleans and converts non-numeric sqft values  
- Removes outliers (sqft/BHK and price_per_sqft)  
- Manually performs **One-Hot Encoding**  
- Trains a **Linear Regression** model  
- Saves model + encoder for deployment  
- Provides a **Streamlit UI** with input fields and real-time predictions  

---

## 📂 Dataset Description

The dataset contains:

| Column        | Description |
|---------------|-------------|
| `area_type`   | Built-up / Super built-up / Plot (dropped) |
| `availability`| Ready to move / Year (dropped) |
| `location`    | Locality in Bengaluru |
| `size`        | Example: "2 BHK", "4 Bedroom" |
| `society`     | Society/complex (dropped) |
| `total_sqft`  | Total square footage (cleaned extensively) |
| `bath`        | Number of bathrooms |
| `balcony`     | Number of balconies |
| `price`       | House price in Lakhs |

Columns dropped: `area_type`, `availability`, `society`  
Reason: Too many missing values or low predictive power.

---

## 🧹 Data Cleaning & Feature Engineering

### ✔ Major cleaning steps:
- Removed rows with missing essential values  
- Converted `size` → numeric **BHK**  
- Cleaned `total_sqft` (handled ranges like `"1200-1500"` → 1350)  
- Created **price_per_sqft**  
- Standardized location names  
- Grouped rare locations into “other”  
- Removed unrealistic sqft/BHK values (< 300)  
- Removed outliers using **IQR within each location**

---

## 📊 Exploratory Data Analysis (Optional)

Visualizations include:

- Price distribution  
- Sqft vs Price scatterplot  
- Bathroom vs Price boxplot  
- Correlation heatmaps  

These help understand the real estate market patterns in Bengaluru.

---

## 🧠 Machine Learning Model

### ✔ Features used:
- `location` (OneHotEncoded manually)
- `total_sqft`
- `bath`
- `balcony`
- `bhk`

### ✔ Target:
- `price` (in Lakhs)

Model used:


### ✔ Evaluation Metrics:
- **RMSE (Root Mean Square Error)**
- **R² Score**
- **Cross-validation**

Typical performance:
- RMSE ≈ 10–20 Lakhs  
- R² ≈ 0.80–0.88  

---

## 💾 Saving the Model (Manual Artifacts)

```python
model_artifacts = {
    "regressor": regressor,
    "encoder": ohe,
    "numeric_columns": ['total_sqft', 'bath', 'balcony', 'bhk']
}

pickle.dump(model_artifacts, open("bangalore_house_price_model.pkl", "wb"))
