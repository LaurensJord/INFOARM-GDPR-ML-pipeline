"""
Configuration file for the GDPR Privacy Policy Classification Research
Based on Okonicha and Sadovykh (2025) mapping of OPP-115 categories to GDPR Articles 13/14
"""

import torch

# =============================================================================
# RANDOM SEEDS AND REPRODUCIBILITY
# =============================================================================
RANDOM_SEED = 42

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================
MAX_LEN = 160
DROPOUT_PROBABILITY = 0.3
EPOCHS_LIST = [3, 5, 10]
BATCH_SIZE_LIST = [16, 32]
LEARNING_RATE_LIST = [2e-5, 3e-5]

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# =============================================================================
# OPP-115 CATEGORIES (from Wilson et al., 2016)
# =============================================================================
OPP_115_CATEGORIES = {
    "First Party Collection/Use": 0,
    "Third Party Sharing/Collection": 1, 
    "User Choice/Control": 2,
    "User Access, Edit, & Deletion": 3,
    "Data Retention": 4,
    "Data Security": 5,
    "Policy Change": 6,
    "Do Not Track": 7,
    "International & Specific Audiences": 8,
    "Other": 9
}

# =============================================================================
# GDPR ARTICLES 13 AND 14 PROVISIONS
# Based on Okonicha and Sadovykh (2025) mapping
# =============================================================================
GDPR_PROVISIONS = {
    # Article 13(1) - Information to be provided where personal data are collected from the data subject
    "Art13_1_a": "Identity and contact details of the controller",
    "Art13_1_b": "Contact details of the data protection officer",
    "Art13_1_c": "Purposes of processing and legal basis",
    "Art13_1_d": "Legitimate interests pursued",
    "Art13_1_e": "Recipients or categories of recipients",
    "Art13_1_f": "Transfer to third country intention",
    
    # Article 13(2) - Additional information for fair and transparent processing
    "Art13_2_a": "Data retention period or criteria",
    "Art13_2_b": "Right to access, rectification, erasure",
    "Art13_2_c": "Right to withdraw consent",
    "Art13_2_d": "Right to lodge a complaint",
    "Art13_2_e": "Statutory or contractual requirement",
    "Art13_2_f": "Existence of automated decision-making",
    
    # Article 14 - Information to be provided where data not obtained from data subject
    "Art14_1": "Source of personal data",
    "Art14_2": "Categories of personal data",
    
    # Additional relevant provisions
    "Art5_data_minimization": "Data minimization principle",
    "Art6_legal_basis": "Legal basis for processing",
    
    # Catch-all
    "other": "Not directly related to GDPR transparency requirements"
}

# =============================================================================
# OPP-115 TO GDPR MAPPING
# Based on Okonicha and Sadovykh (2025) methodology
# =============================================================================
OPP_TO_GDPR_MAPPING = {
    "First Party Collection/Use": [
        "Art13_1_c",  # Purposes of processing
        "Art13_1_d",  # Legitimate interests
        "Art14_2"     # Categories of personal data
    ],
    "Third Party Sharing/Collection": [
        "Art13_1_e",  # Recipients or categories of recipients
        "Art13_1_f"   # Transfer to third country
    ],
    "User Choice/Control": [
        "Art13_2_c",  # Right to withdraw consent
        "Art13_2_b"   # Right to access, rectification, erasure
    ],
    "User Access, Edit, & Deletion": [
        "Art13_2_b"   # Right to access, rectification, erasure
    ],
    "Data Retention": [
        "Art13_2_a"   # Data retention period
    ],
    "Data Security": [
        "Art5_data_minimization"  # Data security principles
    ],
    "Policy Change": [
        "other"       # Policy changes not directly GDPR provision
    ],
    "Do Not Track": [
        "Art13_2_c"   # Related to consent and tracking choices
    ],
    "International & Specific Audiences": [
        "Art13_1_f"   # Transfer to third country
    ],
    "Other": [
        "other"
    ]
}

# =============================================================================
# SIMPLIFIED GDPR CLASS MAPPING FOR MULTI-CLASS CLASSIFICATION
# =============================================================================
GDPR_CLASS_MAPPING = {
    "Purpose": 0,           # Art 13(1)(c) - Purposes of processing
    "Recipients": 1,        # Art 13(1)(e) - Recipients of data
    "DataSubjectRights": 2, # Art 13(2)(b,c) - User rights
    "Retention": 3,         # Art 13(2)(a) - Data retention
    "Security": 4,          # Art 5 - Data security
    "ThirdParty": 5,        # Art 13(1)(e,f) - Third party sharing
    "Transfer": 6,          # Art 13(1)(f) - International transfer
    "other": 7              # Not directly GDPR related
}

# Reverse mapping for labels
GDPR_CLASS_NAMES = {v: k for k, v in GDPR_CLASS_MAPPING.items()}

# =============================================================================
# OPP-115 TO SIMPLIFIED GDPR MAPPING
# =============================================================================
OPP_TO_SIMPLIFIED_GDPR = {
    "First Party Collection/Use": "Purpose",
    "Third Party Sharing/Collection": "ThirdParty", 
    "User Choice/Control": "DataSubjectRights",
    "User Access, Edit, & Deletion": "DataSubjectRights",
    "Data Retention": "Retention",
    "Data Security": "Security",
    "Policy Change": "other",
    "Do Not Track": "DataSubjectRights",
    "International & Specific Audiences": "Transfer",
    "Other": "other"
}

# =============================================================================
# MODEL PATHS AND PRETRAINED MODELS
# =============================================================================
PRE_TRAINED_MODELS = {
    "BERT": "bert-base-uncased",
    "RoBERTa": "roberta-base",
}

# Sentence Transformer for embeddings
ST_MODEL = "sentence-transformers/paraphrase-mpnet-base-v2"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# BiLSTM configuration
BILSTM_CONFIG = {
    "input_size": 384,
    "hidden_size": 768,
    "num_layers": 2,
    "bidirectional": True,
    "dropout": 0.5
}

# =============================================================================
# GEMMA-3 LLM CONFIGURATION
# =============================================================================
GEMMA_CONFIG = {
    "model_name": "google/gemma-3-2b-it",  # Instruction-tuned version
    "max_new_tokens": 50,
    "temperature": 0.1,
    "do_sample": False,
    "device_map": "auto"
}

# Classification prompt template for Gemma-3
GEMMA_CLASSIFICATION_PROMPT = """You are a GDPR compliance expert. Classify the following privacy policy sentence into one of these GDPR-related categories:

Categories:
- Purpose: Related to purposes of data processing (Article 13(1)(c))
- Recipients: Related to recipients of personal data (Article 13(1)(e))  
- DataSubjectRights: Related to user rights like access, rectification, erasure, consent withdrawal (Article 13(2)(b,c))
- Retention: Related to data retention periods (Article 13(2)(a))
- Security: Related to data security measures (Article 5)
- ThirdParty: Related to third party data sharing (Article 13(1)(e,f))
- Transfer: Related to international data transfers (Article 13(1)(f))
- other: Not directly related to GDPR transparency requirements

Sentence: "{sentence}"

Respond with ONLY the category name, nothing else."""

# =============================================================================
# EVALUATION METRICS
# =============================================================================
METRICS = ["accuracy", "precision", "recall", "f1", "f2"]

# F2 score beta value (emphasizes recall over precision)
F2_BETA = 2

# =============================================================================
# DATA SPLIT CONFIGURATION
# =============================================================================
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

# =============================================================================
# COMPUTATIONAL METRICS CONFIGURATION  
# =============================================================================
TRACK_COMPUTATIONAL_METRICS = True

COMPUTATIONAL_METRICS = {
    "training_time": True,
    "inference_time": True,
    "gpu_memory": True,
    "cpu_usage": True,
    "ram_usage": True,
    "energy_estimation": True
}
