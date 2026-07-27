# Used Car Price Prediction & Market Segmentation

A machine-learning project analysing approximately **55,000 second-hand car sales** to predict vehicle prices and identify meaningful market segments.

The project combines supervised and unsupervised learning, comparing regression, ensemble and neural-network approaches before applying clustering to uncover structure in the used-car market.

## Project Objectives

- Predict used-car prices from vehicle characteristics.
- Compare simple regression models with more powerful machine-learning approaches.
- Identify the features that contribute most strongly to price prediction.
- Segment vehicles into meaningful groups using unsupervised learning.

## Dataset

The dataset contains approximately **55,000 records** with attributes including manufacturer and model, engine size, fuel type, year of manufacture, mileage and sale price.

The raw dataset is not included in this repository. See [`data/README.md`](data/README.md) for setup guidance.

## Data Preparation

The workflow includes checking missing/inconsistent values and duplicates, preparing numerical and categorical variables, one-hot encoding categorical features, standardising numerical variables where required, and splitting data for supervised model training and evaluation.

## Supervised Learning

### Single-Feature Regression

| Feature | Linear R² | Polynomial R² |
|---|---:|---:|
| Engine Size | 0.151 | 0.151 |
| Year of Manufacture | 0.511 | 0.609 |
| Mileage | 0.401 | 0.522 |

Year of manufacture was the strongest individual predictor among these features.

### Multiple Linear Regression

- **MSE:** 89,158,615.76
- **R²:** 0.671

### Random Forest Regressor

Random Forest produced the strongest result:

- **MSE:** 415,044.78
- **R²:** **0.9985**

Feature-importance analysis identified **year of manufacture** and **engine size** as particularly influential predictors.

### Artificial Neural Network

A feed-forward ANN was developed using dense layers, ReLU activation and dropout regularisation.

- **MSE:** 1,006,289.50
- **MAE:** 542.84
- **R²:** **0.9963**

The ANN performed strongly, although Random Forest achieved the best overall predictive performance in this experiment.

## Market Segmentation

### K-Means

The strongest tested feature combination used **year of manufacture and mileage**, with an optimal **k = 3** and silhouette score of approximately **0.4405**.

### Agglomerative Clustering

Hierarchical clustering was also applied using the same feature pair and number of clusters, providing a second view of the dataset's segmentation structure.

## Key Findings

- Combining multiple vehicle characteristics substantially improved price prediction over single-feature regression.
- **Random Forest was the strongest predictive model**, achieving R² ≈ 0.9985.
- The ANN also performed strongly with R² ≈ 0.9963.
- Vehicle age and mileage were useful for identifying natural market segments.
- Supervised and unsupervised learning provided complementary insights into the used-car market.

## Technologies

**Python • Pandas • NumPy • Scikit-learn • TensorFlow • Keras • Matplotlib • Regression • Random Forest • ANN • K-Means • Agglomerative Clustering**

## Repository Structure

```text
used-car-price-prediction/
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
├── notebooks/
│   └── used_car_price_prediction.ipynb
├── data/
│   └── README.md
└── results/
    └── figures/
```

## Installation

```bash
git clone https://github.com/Ape-programmer/used-car-price-prediction.git
cd used-car-price-prediction
python -m venv .venv
pip install -r requirements.txt
```

Place the dataset according to `data/README.md`, then start Jupyter from the repository root.

## Limitations

- Very high predictive scores should be interpreted in the context of the supplied dataset and its feature relationships.
- Results may not generalise directly to current market prices or unseen manufacturers/models.
- K-Means is sensitive to scaling and feature selection.
- The analysis is predictive and exploratory rather than causal.

## Future Improvements

Potential extensions include cross-validation and systematic hyperparameter optimisation, XGBoost/LightGBM comparison, SHAP-based explainability, richer vehicle-age and brand features, robust outlier analysis and deployment as an interactive price-estimation application.

## Author

**Abiola Peace Emmanuel**  
MSc Artificial Intelligence & Data Science  
GitHub: **Ape-programmer**
