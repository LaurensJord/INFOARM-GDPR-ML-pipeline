# GDPR Privacy Policy Classification Pipeline

A multi-solution benchmarking study comparing machine learning approaches for classifying privacy policy sentences against GDPR transparency requirements (Articles 13 & 14).

This repository extends the original DPA completeness checking work to privacy policies using the **OPP-115 dataset** mapped to **GDPR provisions** via the JURIX 2020 methodology.

## Research Objective

Evaluate and compare different ML technologies for privacy policy classification:
- **Traditional ML**: Logistic Regression, SVM, Random Forest
- **Deep Learning**: BiLSTM, MLP
- **Transformers**: BERT, RoBERTa, ALBERT, Legal-BERT
- **Few-shot Learning**: SetFit
- **LLM**: Gemma-3 (planned)

## Quick Start

```bash
# Clone the repository
git clone <repository-url>
cd INFOARM-GDPR-ML-pipeline

# Run setup (creates venv, installs dependencies, downloads datasets)
chmod +x setup.sh
./setup.sh

# Activate the environment
source .venv/bin/activate

# Preprocess OPP-115 for GDPR classification
python Resources/data_preprocessing.py Input/OPP-115

# Run the test notebook
jupyter notebook test_opp115_classification.ipynb
```

## Dataset Pipeline

### 1. OPP-115 Dataset
The [OPP-115 corpus](https://usableprivacy.org/data) contains **115 website privacy policies** with over **23,000 fine-grained annotations** across 10 privacy practice categories:

| OPP-115 Category | Description |
|------------------|-------------|
| First Party Collection/Use | Data collected and used by the service |
| Third Party Sharing/Collection | Data shared with or collected by third parties |
| User Choice/Control | Options for users to control their data |
| User Access, Edit and Deletion | Rights to access, modify, or delete data |
| Data Retention | How long data is stored |
| Data Security | Security measures for data protection |
| Policy Change | How policy changes are communicated |
| Do Not Track | DNT signal handling |
| International and Specific Audiences | Special handling for children, EU users, etc. |
| Other | Introductory text, definitions, etc. |

### 2. JURIX 2020 GDPR Mapping
The [JURIX 2020 paper](https://www.usableprivacy.org/static/files/swilson_acl_2016.pdf) ("From Prescription to Description: Mapping the GDPR to a Privacy Policy Corpus Annotation Scheme") provides the mapping between OPP-115 categories and GDPR articles/principles.

### 3. Simplified Classification Schema
For practical multi-class classification, we map OPP-115 to 9 GDPR-aligned categories:

```
DataCollection      → First Party Collection/Use
ThirdPartySharing   → Third Party Sharing/Collection  
UserRights          → User Choice/Control + Access/Edit/Deletion
DataRetention       → Data Retention
DataSecurity        → Data Security
PolicyChange        → Policy Change
DoNotTrack          → Do Not Track
SpecialAudiences    → International and Specific Audiences
Other               → Other/Introductory
```

### 4. Data Flow

```
OPP-115 Dataset (115 policies, ~23K annotations)
        ↓
JURIX 2020 Mapping (OPP-115 → GDPR Articles/Principles)
        ↓
data_preprocessing.py (consolidation @ 0.75 threshold)
        ↓
opp115_train.csv (14,336 samples) + opp115_test.csv (2,480 samples)
        ↓
Classification Models (BERT, RoBERTa, BiLSTM, SVM, etc.)
        ↓
Metrics: Precision, Recall, F1-Score, F2-Score
```

## Repository Structure

```
├── setup.sh                      # Environment setup & dataset download
├── requirements.txt              # Python dependencies
├── Input/
│   ├── OPP-115/                  # Raw OPP-115 dataset (downloaded)
│   │   ├── consolidation/        # Majority-voted annotations
│   │   ├── pretty_print/         # CSV format annotations
│   │   └── sanitized_policies/   # Clean HTML policies
│   ├── jurix_2020_opp-115_gdpr_dataset/  # GDPR mapping matrices
│   ├── opp115_train.csv          # Preprocessed training data
│   └── opp115_test.csv           # Preprocessed test data
├── Resources/
│   ├── config.py                 # GDPR provisions & model configs
│   ├── data_preprocessing.py     # OPP-115 → GDPR preprocessing
│   ├── utils.py                  # Model training/evaluation utilities
│   └── pretrained_models/        # Saved model checkpoints
├── test_opp115_classification.ipynb  # Quick test with baseline models
├── train_model.ipynb             # Transformer model training
├── test_models.ipynb             # Model evaluation
└── baseline_models.ipynb         # Traditional ML & deep learning
```

## Requirements

- Python 3.10+
- PyTorch 2.0+
- Transformers 4.30+
- scikit-learn, pandas, numpy
- See `requirements.txt` for full list

## License

The base of this code is developed at SnT / University of Luxembourg with funding from Luxembourg's National Research Fund (FNR).

The code was adopted for an experiment ran by students from the Utrecht University.






