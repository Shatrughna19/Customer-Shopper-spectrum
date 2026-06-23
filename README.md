# Shopper Spectrum 🛒

**Customer Segmentation & Product Recommendation Engine**

A comprehensive machine learning solution for e-commerce customer intelligence. Uses RFM analysis and KMeans clustering to segment customers into 4 behavioral spectrums, then recommends similar products via collaborative filtering.

---

## 🎯 Features

- **📊 RFM-based Customer Segmentation** — Analyze Recency, Frequency, and Monetary metrics
- **🎯 KMeans Clustering** — Predict customer spectrum segments for new buyers
- **🔍 Product Recommender** — Find top-5 similar products using cosine similarity
- **📈 Interactive Streamlit Dashboard** — Visualize insights and make real-time predictions
- **📉 Comprehensive EDA** — Explore dataset statistics, trends, and correlations
- **💾 Persistent Models** — Pre-trained models ready for inference

---

## 📁 Project Structure

```
final/
├── app.py                              # Streamlit main entry point
├── pages/
│   ├── 1_Home.py                       # Dataset overview & statistics
│   ├── 2_Clustering.py                 # RFM segmentation predictor
│   └── 3_Recommendation.py             # Product similarity recommender
├── CustomerEDA.ipynb                   # Exploratory Data Analysis notebook
├── train_spectrum_model.py             # Model training                        script                   # Python dependencies
├── data/
│   ├── online_retail_cleaned.csv       # Cleaned transaction data
│   ├── rfm_features.csv                # RFM metrics for all customers
│   ├── customer_product_matrix.csv     # Customer-Product interaction matrix
│   └── rfm_with_segments.csv           # RFM with assigned clusters
├── models/
│   ├── rfm_scaler.pkl                  # StandardScaler for RFM normalization
│   ├── kmeans_model.pkl                # Trained KMeans clustering model
│   ├── cluster_labels.json             # Mapping: cluster ID → segment name
│   ├── similarity_matrix.pkl           # Pre-computed cosine similarity matrix
│   └── top_recommendations.json        # Pre-computed recommendations
└── figures/                            # Output visualizations
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip or conda
- Virtual environment (recommended)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/shopper-spectrum.git
cd shopper-spectrum
```

2. **Create virtual environment**
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

---

## 📖 Usage

### Step 1: Run Exploratory Data Analysis

Open and execute the Jupyter notebook to clean data and compute RFM features:

```bash
jupyter notebook CustomerEDA.ipynb
```

**What it does:**
- Loads raw e-commerce dataset
- Cleans data (removes cancelled invoices, missing values, duplicates)
- Performs exploratory analysis with visualizations
- Computes RFM metrics for all customers
- Exports cleaned data for model training

**Outputs:**
- `data/online_retail_cleaned.csv`
- `data/rfm_features.csv`
- `data/customer_product_matrix.csv`

---

### Step 2: Train Clustering & Recommendation Models

```bash
python train_spectrum_model.py
```

**What it does:**
- Loads RFM features from EDA
- Scales RFM metrics using StandardScaler
- Trains KMeans clustering (4 clusters)
- Labels clusters as: High-Value, Regular, Occasional, At-Risk
- Computes cosine similarity matrix for product recommendations
- Saves all models and matrices for production use

**Outputs:**
- `models/rfm_scaler.pkl` — Scaler for new predictions
- `models/kmeans_model.pkl` — Trained clustering model
- `models/cluster_labels.json` — Segment names
- `models/similarity_matrix.pkl` — Product recommendations
- `data/rfm_with_segments.csv` — Full customer segments

**Training Time:** ~2-5 minutes

---

### Step 3: Launch Interactive Dashboard

```bash
streamlit run app.py
```

Opens interactive web app at `http://localhost:8501`

#### **Page 1: Home 🏠**
- Dataset statistics (transaction count, unique customers, etc.)
- Geographic distribution
- Segment distribution chart
- Quick-start instructions

#### **Page 2: Clustering 🎯**
- Input customer RFM values
- Get predicted spectrum segment
- View segment characteristics & recommendations
- See scaled features for transparency

#### **Page 3: Recommendation 🔍**
- Enter a product name
- Get top-5 similar products
- View similarity scores
- Browse product catalog

---

## 📊 Customer Segments Explained

| Segment | Recency | Frequency | Monetary | Action |
|---------|---------|-----------|----------|--------|
| **High-Value** | Low (recent) | High | High | Loyalty rewards, VIP offers |
| **Regular** | Medium | Medium | Medium | Upsell, cross-sell campaigns |
| **Occasional** | Medium-High | Low | Low | Re-engagement emails |
| **At-Risk** | High (old) | Low | Low-Med | Win-back discounts |

---

## 📈 RFM Metrics

- **Recency (R):** Days since last purchase → *Lower is better (more engaged)*
- **Frequency (F):** Number of unique orders → *Higher is better (loyal customer)*
- **Monetary (M):** Total lifetime spend → *Higher is better (valuable customer)*

### Example:
- Customer A: R=10 days, F=15 orders, M=£2,500 → **High-Value** ✅
- Customer B: R=200 days, F=2 orders, M=£150 → **At-Risk** ⚠️

---

## 🛠 Technologies Used

| Component | Library | Purpose |
|-----------|---------|---------|
| **Data Processing** | Pandas, NumPy | EDA, feature engineering |
| **ML & Clustering** | Scikit-learn | RFM scaling, KMeans, similarity |
| **Visualization** | Matplotlib, Seaborn | EDA charts and analysis |
| **Web App** | Streamlit | Interactive dashboard |
| **Model Persistence** | Joblib, Pickle | Save/load trained models |

---

## 📦 Dependencies

```
pandas>=2.1.0
numpy>=1.26.0
scikit-learn>=1.3.0
matplotlib>=3.8.0
seaborn>=0.13.0
joblib>=1.3.0
streamlit>=1.28.0
```

Install via: `pip install -r requirements.txt`

---

## 📁 Dataset Information

**Source:** UCI Machine Learning Repository - Online Retail Dataset

**Size:** ~500K transactions, ~4K customers, ~4K products

**Time Period:** Dec 2010 - Dec 2011

**Geographic Coverage:** 38 countries (UK-dominant)

**Data Cleaning Applied:**
- ✅ Removed cancelled invoices
- ✅ Dropped missing CustomerID rows
- ✅ Removed negative/zero quantities and prices
- ✅ Eliminated duplicate rows
- ✅ Converted data types (dates, integers)

---

## 🎓 Analysis Insights

### Key Findings from EDA

1. **Geographic Concentration:** UK accounts for ~80% of revenue
2. **Product Pareto:** Top 15 products drive ~40% of revenue
3. **Seasonality:** Strong peaks during Nov-Dec holiday season
4. **Customer Distribution:** Right-skewed; most customers are low-value
5. **RFM Correlation:** Frequency ↔ Monetary (~0.7), Recency ↔ Frequency (~-0.5)

### Segmentation Results

- **High-Value:** ~8% of customers, ~60% of revenue
- **Regular:** ~15% of customers, ~25% of revenue
- **Occasional:** ~35% of customers, ~10% of revenue
- **At-Risk:** ~42% of customers, ~5% of revenue

---

## 🚦 Model Performance

| Metric | Value |
|--------|-------|
| Silhouette Score | 0.52 |
| Within-Cluster Sum of Squares | ~85K |
| Cluster Separation | Good |

---

## 📝 How to Make Predictions

### Python API (Using Trained Models)

```python
import joblib
import numpy as np

# Load models
scaler = joblib.load('models/rfm_scaler.pkl')
kmeans = joblib.load('models/kmeans_model.pkl')

# New customer RFM values
recency, frequency, monetary = 45, 8, 500.0

# Scale and predict
X = np.array([[recency, frequency, monetary]])
X_scaled = scaler.transform(X)
cluster = kmeans.predict(X_scaled)[0]

print(f"Customer Cluster: {cluster}")
```

---







---





**Last Updated:** June 2026  

