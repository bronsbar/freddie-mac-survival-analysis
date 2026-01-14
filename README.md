# Freddie Mac Survival Analysis

Survival analysis and machine learning techniques applied to Freddie Mac Single Family Loan data for modeling mortgage default and prepayment risks.

## Project Overview

This project applies survival analysis techniques to predict time-to-event outcomes in mortgage loans:
- **Default**: Loan becomes seriously delinquent (90+ days)
- **Prepayment**: Borrower pays off loan early
- **Censored**: Loan is still active (right-censored)

### Competing Risks Framework

Mortgage loans have competing risks where default and prepayment are mutually exclusive terminal events. This project implements:
- Cause-specific hazard models
- Cumulative incidence functions
- ML-based survival models (Random Survival Forests, DeepSurv, etc.)

## Data Source

### Freddie Mac Single Family Loan-Level Dataset

**Official source**: https://www.freddiemac.com/research/datasets/sf-loanlevel-dataset

The dataset contains:
- **Origination data**: Static loan characteristics at origination
- **Monthly performance data**: Dynamic loan status tracking

#### Key Variables for Survival Analysis

| Variable | Description |
|----------|-------------|
| `loan_age` | Months since origination (time variable) |
| `zero_balance_code` | Event type: 01=Prepaid, 03=Short Sale, 09=REO, etc. |
| `current_loan_delinquency_status` | Delinquency indicator |
| `zero_balance_effective_date` | Date of terminal event |

#### Sample Dataset

A sample dataset of 50,000 loans per vintage year is available for initial exploration.

### Alternative Data Sources

- [Kaggle: Fannie Mae & Freddie Mac Database 2008-2018](https://www.kaggle.com/datasets/jeromeblanchet/fannie-mae-freddie-mac-public-use-database)
- [Kaggle: Freddie Mac Pre-processed](https://www.kaggle.com/datasets/nikunjhemani/freddie-macs-dataset-pre-processed)

## Project Structure

```
freddie-mac-survival-analysis/
├── data/
│   ├── raw/                 # Original Freddie Mac data files
│   ├── processed/           # Cleaned survival analysis-ready data
│   └── external/            # External datasets (macroeconomic indicators, etc.)
├── notebooks/               # Jupyter notebooks for exploration and analysis
├── src/
│   ├── data/               # Data loading and preprocessing scripts
│   ├── features/           # Feature engineering for survival models
│   ├── models/             # Model implementations
│   └── visualization/      # Plotting functions for survival curves
├── reports/
│   └── figures/            # Generated graphics and figures
├── tests/                  # Unit tests
├── requirements.txt        # Python dependencies
└── README.md
```

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/freddie-mac-survival-analysis.git
cd freddie-mac-survival-analysis

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Data Preprocessing

The raw Freddie Mac data must be transformed into survival analysis format:

```python
# Example survival data structure
# loan_id | duration | event | event_type | covariates...
# 001     | 36       | 1     | default    | ltv=80, fico=720, ...
# 002     | 48       | 1     | prepay     | ltv=75, fico=780, ...
# 003     | 24       | 0     | censored   | ltv=90, fico=680, ...
```

Run preprocessing:
```bash
python src/data/preprocess.py --input data/raw --output data/processed
```

## Models Implemented

### Classical Survival Analysis
- Kaplan-Meier estimator
- Cox Proportional Hazards model
- Accelerated Failure Time (AFT) models

### Machine Learning Approaches
- Random Survival Forests
- Gradient Boosted Survival Analysis
- DeepSurv (neural network)
- Cox-Time

### Competing Risks Models
- Cause-specific Cox models
- Fine-Gray subdistribution hazard model

## Usage

```python
from src.models import CoxPH, RandomSurvivalForest
from src.data import load_processed_data

# Load data
X, duration, event = load_processed_data('data/processed/survival_data.csv')

# Fit Cox model
cox = CoxPH()
cox.fit(X, duration, event)

# Predict survival probabilities
surv_probs = cox.predict_survival_function(X_test)
```

## References

- Deng, Y. (1997). Mortgage termination: An empirical hazard model with a stochastic term structure.
- Stepanova, M., & Thomas, L. (2002). Survival analysis methods for personal loan data.
- Katzman et al. (2018). DeepSurv: personalized treatment recommender system using a Cox proportional hazards deep neural network.

## License

This project is for educational and research purposes.
