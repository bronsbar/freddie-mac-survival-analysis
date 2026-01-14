# Project Plan: Freddie Mac Survival Analysis

## Phase 1: Data Acquisition
1. **Register at Freddie Mac** - Create account at freddiemac.com/research/datasets/sf-loanlevel-dataset
2. **Download sample dataset** - Get the 50,000 loans per vintage year sample
3. **Download data dictionary** - Understand all variable definitions
4. **Store raw files** in `data/raw/`

## Phase 2: Data Preprocessing
1. **Parse origination files** - Extract static loan characteristics (LTV, FICO, loan amount, etc.)
2. **Parse performance files** - Extract monthly loan status records
3. **Merge datasets** - Join origination and performance by loan ID
4. **Create survival format**:
   - Calculate `duration` (loan age in months)
   - Define `event` indicator (0=censored, 1=event occurred)
   - Define `event_type` (default, prepayment)
   - Map `zero_balance_code` to event types
5. **Handle missing values** and data quality issues
6. **Output to** `data/processed/`

## Phase 3: Exploratory Data Analysis
1. **Univariate analysis** of key variables
2. **Kaplan-Meier curves** for overall survival
3. **Cumulative incidence functions** for competing risks
4. **Stratified analysis** by FICO buckets, LTV ranges, vintage years

## Phase 4: Feature Engineering
1. **Static features**: credit score bands, LTV categories, loan purpose, property type
2. **Time-varying features**: current delinquency status, interest rate changes
3. **Macroeconomic features**: unemployment rate, HPI changes (optional, external data)

## Phase 5: Model Implementation
1. **Classical models**:
   - Kaplan-Meier estimator
   - Cox Proportional Hazards
   - AFT models
2. **ML models**:
   - Random Survival Forests
   - Gradient Boosted Survival
3. **Competing risks models**:
   - Cause-specific Cox
   - Fine-Gray model
4. **Deep learning** (optional):
   - DeepSurv / Cox-Time

## Phase 6: Model Evaluation
1. **Concordance index** (C-statistic)
2. **Brier score** for calibration
3. **Time-dependent AUC**
4. **Cross-validation** for model comparison

## Phase 7: Documentation & Visualization
1. **Survival curves** by risk factors
2. **Hazard ratio forest plots**
3. **Model comparison tables**
4. **Final report** with findings
