# 🏠 Housing Price Analytics & Prediction  
### by **Unmesh Bajirao Dighe**

An end-to-end **Machine Learning and Data Analytics project** that predicts housing prices and provides interactive visual insights through a **Streamlit web application**.

This project demonstrates skills in:
✅ Data preprocessing  
✅ Feature engineering  
✅ Model training & evaluation  
✅ Deployment-ready web app  
✅ Real-world dataset handling  

---

## 📌 Project Features

### 🔮 Housing Price Prediction
- Predicts price based on 20+ property attributes
- Uses trained **Random Forest Regression** model
- Provides:
  ✅ Price estimate in ₹  
  ✅ Category (Budget / Mid-Range / Premium)  
  ✅ Price percentile  
  ✅ Price vs average comparison  
  ✅ Price per sqft  

### 📊 Interactive Analytics Dashboard
Includes:
- Price distribution visualization  
- Feature correlation heatmap  
- Avg price by bedrooms  
- Avg price by construction grade  

### 📋 Dataset Exploration
- View first 100 records  
- Summary statistics  
- Total samples + feature count  

---

## 📂 Project Structure

```text
Housing Price Analytics/
│
├─ data/
│  └─ Housing.csv
│
├─ models/
│  ├─ housing_price_model.pkl
│  └─ model_features.pkl
│
├─ src/
│  ├─ train_model.py
│  └─ predict_example.py
│
├─ app.py
└─ README.md
```

---

## 🧠 Dataset Overview

✅ 14,620 housing records  
✅ 23 meaningful real-estate features  

Example processed columns:

| Feature | Description |
|---------|-------------|
| number_of_bedrooms | Bedroom count |
| number_of_bathrooms | Bathrooms |
| living_area | Sqft interior |
| lot_area | Land area |
| grade_of_the_house | Build quality |
| built_year | Construction year |
| renovation_year | Last renovation |
| postal_code | Location indicator |
| lattitude / longitude | Geo coordinates |
| price | **TARGET VALUE** |

---

## 🛠 Tech Stack

### **Backend / ML**
- Python 3.x  
- Pandas, NumPy  
- Scikit-Learn  
- Joblib  

### **Frontend / Visualization**
- Streamlit  
- Plotly  

---

## ⚙️ Setup Instructions

### 1️⃣ Create Virtual Environment
```bash
python -m venv .venv
```

Activate it:

**Windows:**
```bash
.venv\Scriptsctivate
```

### 2️⃣ Install Dependencies
```bash
pip install streamlit pandas numpy scikit-learn joblib plotly
```

---

## 🧪 Train the Model

From project root:

```bash
python src/train_model.py
```

This will:

✅ Load & clean dataset  
✅ One-hot encode postal_code  
✅ Train Random Forest model  
✅ Save:

```
models/housing_price_model.pkl
models/model_features.pkl
```

---

## 🚀 Run the Streamlit App

```bash
streamlit run app.py
```

Then open browser (auto or):

```
http://localhost:8501
```

---

## 📈 Example Output Screens

✅ Price Estimation Box  
✅ Similar Property Suggestions  
✅ Price Analytics Graphs  
✅ Dataset Statistics  

---

## 📝 Resume-Ready Highlights (You Can Use)

- Developed an end-to-end **Housing Price Prediction System** using Python and Machine Learning.
- Cleaned and engineered features from a real dataset with **14,620+ samples**.
- Trained a **Random Forest Regression** model achieving strong MAE/RMSE performance.
- Built an interactive **Streamlit web app** for real-time price prediction and analytics.
- Added advanced features like **price percentile**, **comparative analysis**, and **similar property recommendations**.

---

## 🚀 Future Enhancements

✅ Hyperparameter tuning  
✅ XGBoost / Gradient Boosting comparison  
✅ Deployment on Render / HuggingFace / AWS  
✅ Live real-estate API integration  

---

## 👤 Author

### **Unmesh Bajirao Dighe**
📍 Computer Engineering  
💡 Data Analyst & Machine Learning Enthusiast  
📧 (Add your email here if desired)

---

## ⭐ How to Use This Project

✅ Add to GitHub portfolio  
✅ Showcase during interviews  
✅ Mention in resume  
✅ Use as base for future ML apps  

---

## ✅ End of README
"# Housing-Price-Analytics-and-Prediction" 
