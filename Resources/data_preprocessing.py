"""
Data Preprocessing for OPP-115 Dataset with GDPR Mapping
Based on JURIX 2020 paper: "From Prescription to Description: Mapping the GDPR to a Privacy Policy Corpus Annotation Scheme"

This script processes the OPP-115 dataset and maps categories to GDPR Articles/Principles
for the privacy policy classification research.
"""

import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")

# =============================================================================
# GDPR MAPPING FROM JURIX 2020 DATASET
# =============================================================================

# OPP-115 Categories to GDPR Articles mapping (from connections_overview.csv)
OPP_TO_GDPR_ARTICLES = {
    "First Party Collection/Use": [4, 5, 6, 7, 8, 9, 10, 11, 24, 25, 30, 33, 34, 35, 36, 37, 38, 39, 89, 91, 95],
    "Third Party Sharing/Collection": [4, 6, 9, 19, 28, 29, 30, 37, 38, 39, 44, 45, 46, 47, 48, 49, 96],
    "User Choice/Control": [4, 6, 7, 8, 9, 13, 14, 17, 18, 20, 21, 26, 49, 77, 78, 79, 80, 82],
    "User Access, Edit and Deletion": [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 25],
    "Data Retention": [5, 25, 30],
    "Data Security": [4, 5, 6, 12, 24, 25, 28, 30, 32, 33, 34, 35, 36, 45, 89],
    "Policy Change": [],  # None
    "Do Not Track": [],   # None
    "International and Specific Audiences": [8],
    "Other": []
}

# OPP-115 Categories to GDPR Principles mapping
OPP_TO_GDPR_PRINCIPLES = {
    "First Party Collection/Use": ["Lawfulness_fairness_transparency", "Purpose_limitation", "Data_minimisation"],
    "Third Party Sharing/Collection": ["Lawfulness_fairness_transparency", "Purpose_limitation", "Data_minimisation"],
    "User Choice/Control": ["Lawfulness_fairness_transparency"],
    "User Access, Edit and Deletion": ["Lawfulness_fairness_transparency", "Accuracy"],
    "Data Retention": ["Storage_limitation"],
    "Data Security": ["Integrity_confidentiality"],
    "Policy Change": ["Lawfulness_fairness_transparency"],
    "Do Not Track": [],
    "International and Specific Audiences": ["Lawfulness_fairness_transparency"],
    "Other": []
}

# Simplified category mapping for multi-class classification
# Based on Articles 13 and 14 transparency requirements
SIMPLIFIED_GDPR_MAPPING = {
    "First Party Collection/Use": "DataCollection",       # Art 13/14 - purposes of processing
    "Third Party Sharing/Collection": "ThirdPartySharing", # Art 13(1)(e,f) - recipients
    "User Choice/Control": "UserRights",                  # Art 13(2)(b,c) - user rights
    "User Access, Edit and Deletion": "UserRights",       # Art 13(2)(b) - access/rectification
    "Data Retention": "DataRetention",                    # Art 13(2)(a) - retention period
    "Data Security": "DataSecurity",                      # Art 5(1)(f) - security
    "Policy Change": "PolicyChange",                      # Transparency
    "Do Not Track": "DoNotTrack",                         # User choice
    "International and Specific Audiences": "SpecialAudiences",  # Art 8 - children
    "Other": "Other"
}

# Class labels for multi-class classification
CLASS_LABELS = {
    "DataCollection": 0,
    "ThirdPartySharing": 1,
    "UserRights": 2,
    "DataRetention": 3,
    "DataSecurity": 4,
    "PolicyChange": 5,
    "DoNotTrack": 6,
    "SpecialAudiences": 7,
    "Other": 8
}

CLASS_NAMES = {v: k for k, v in CLASS_LABELS.items()}


def load_opp115_annotations(base_path):
    """
    Load OPP-115 annotations from the consolidated folder.
    
    Args:
        base_path: Path to OPP-115 dataset
    
    Returns:
        DataFrame with all annotations
    """
    annotations_path = os.path.join(base_path, "OPP-115", "annotations")
    
    all_data = []
    
    # Load each policy's annotations
    for policy_folder in os.listdir(annotations_path):
        policy_path = os.path.join(annotations_path, policy_folder)
        if os.path.isdir(policy_path):
            # Load the JSON annotation files
            for file in os.listdir(policy_path):
                if file.endswith('.json'):
                    with open(os.path.join(policy_path, file), 'r') as f:
                        try:
                            data = json.load(f)
                            # Process annotations
                            for annotation in data.get('annotations', []):
                                all_data.append({
                                    'policy_id': policy_folder,
                                    'annotator': file.replace('.json', ''),
                                    'text': annotation.get('selectedText', ''),
                                    'category': annotation.get('category', 'Other'),
                                    'attributes': annotation.get('attributes', {})
                                })
                        except json.JSONDecodeError:
                            continue
    
    return pd.DataFrame(all_data)


def load_opp115_pretty_print(base_path):
    """
    Load OPP-115 from pretty_print CSV files (simpler format).
    
    Args:
        base_path: Path to OPP-115 dataset
    
    Returns:
        DataFrame with all policy segments
    """
    pretty_print_path = os.path.join(base_path, "OPP-115", "pretty_print")
    
    all_data = []
    
    for csv_file in os.listdir(pretty_print_path):
        if csv_file.endswith('.csv'):
            policy_name = csv_file.replace('.csv', '')
            file_path = os.path.join(pretty_print_path, csv_file)
            
            try:
                # Read CSV without headers as the format varies
                df = pd.read_csv(file_path, header=None, encoding='utf-8', on_bad_lines='skip')
                
                for idx, row in df.iterrows():
                    if len(row) >= 4:
                        all_data.append({
                            'annotation_id': row[0] if pd.notna(row[0]) else idx,
                            'segment_id': row[1] if len(row) > 1 and pd.notna(row[1]) else idx,
                            'policy_id': row[2] if len(row) > 2 and pd.notna(row[2]) else policy_name,
                            'text': row[3] if len(row) > 3 and pd.notna(row[3]) else '',
                            'policy_name': policy_name
                        })
            except Exception as e:
                print(f"Error loading {csv_file}: {e}")
                continue
    
    return pd.DataFrame(all_data)


def load_opp115_consolidation(base_path, threshold="threshold-0.75-overlap-similarity"):
    """
    Load OPP-115 from consolidation folder (majority-voted annotations).
    Uses threshold-0.75-overlap-similarity by default (balanced consensus).
    
    Args:
        base_path: Path to OPP-115 dataset
        threshold: Which consolidation threshold to use:
            - "threshold-0.5-overlap-similarity" (more lenient)
            - "threshold-0.75-overlap-similarity" (balanced)
            - "threshold-1.0-overlap-similarity" (strict)
    
    Returns:
        DataFrame with consolidated annotations
    """
    # Check for nested OPP-115 folder structure
    if os.path.exists(os.path.join(base_path, "OPP-115", "consolidation")):
        consolidation_path = os.path.join(base_path, "OPP-115", "consolidation", threshold)
    else:
        consolidation_path = os.path.join(base_path, "consolidation", threshold)
    
    if not os.path.exists(consolidation_path):
        # Try without OPP-115 subfolder
        consolidation_path = os.path.join(base_path, "consolidation", threshold)
    
    print(f"Loading from: {consolidation_path}")
    
    all_data = []
    
    for csv_file in os.listdir(consolidation_path):
        if csv_file.endswith('.csv'):
            # Extract policy name from filename like "105_amazon.com.csv"
            parts = csv_file.replace('.csv', '').split('_', 1)
            policy_id = parts[0] if len(parts) > 0 else csv_file
            policy_name = parts[1] if len(parts) > 1 else csv_file.replace('.csv', '')
            
            file_path = os.path.join(consolidation_path, csv_file)
            
            try:
                # Read CSV - the consolidation files don't have headers
                df = pd.read_csv(file_path, header=None, encoding='utf-8', on_bad_lines='skip')
                
                # Column structure based on inspection:
                # 0: annotation_id, 1: annotator, 2: policy_id_numeric, 3: segment_id, 
                # 4: paragraph_id, 5: category, 6: attributes_json, 7: date, 8: url
                
                processed_rows = []
                for idx, row in df.iterrows():
                    if len(row) >= 6:
                        # Extract category (column 5)
                        category = str(row[5]) if pd.notna(row[5]) else 'Other'
                        
                        # Extract text from attributes JSON (column 6)
                        text = ""
                        if len(row) > 6 and pd.notna(row[6]):
                            try:
                                attrs = json.loads(str(row[6]))
                                # Get selectedText from any attribute
                                for key, value in attrs.items():
                                    if isinstance(value, dict) and 'selectedText' in value:
                                        text = value.get('selectedText', '')
                                        if text:
                                            break
                            except (json.JSONDecodeError, TypeError):
                                pass
                        
                        if text:  # Only add if we have text
                            processed_rows.append({
                                'annotation_id': row[0] if pd.notna(row[0]) else f"{policy_id}_{idx}",
                                'annotator': row[1] if pd.notna(row[1]) else 'unknown',
                                'policy_id_numeric': row[2] if len(row) > 2 and pd.notna(row[2]) else policy_id,
                                'segment_id': row[3] if len(row) > 3 and pd.notna(row[3]) else idx,
                                'paragraph_id': row[4] if len(row) > 4 and pd.notna(row[4]) else 0,
                                'category': category,
                                'text': text,
                                'policy_name': policy_name,
                                'policy_id': policy_id
                            })
                
                if processed_rows:
                    all_data.extend(processed_rows)
                    
            except Exception as e:
                print(f"Error loading {csv_file}: {e}")
                continue
    
    if all_data:
        result_df = pd.DataFrame(all_data)
        print(f"Loaded {len(result_df)} annotations from {len(set(result_df['policy_name']))} policies")
        return result_df
    return pd.DataFrame()


def map_to_gdpr_category(opp_category):
    """
    Map OPP-115 category to simplified GDPR category.
    
    Args:
        opp_category: OPP-115 category name
    
    Returns:
        Simplified GDPR category
    """
    return SIMPLIFIED_GDPR_MAPPING.get(opp_category, "Other")


def map_to_gdpr_articles(opp_category):
    """
    Map OPP-115 category to relevant GDPR articles.
    
    Args:
        opp_category: OPP-115 category name
    
    Returns:
        List of relevant GDPR article numbers
    """
    return OPP_TO_GDPR_ARTICLES.get(opp_category, [])


def map_to_gdpr_principles(opp_category):
    """
    Map OPP-115 category to relevant GDPR principles.
    
    Args:
        opp_category: OPP-115 category name
    
    Returns:
        List of relevant GDPR principles
    """
    return OPP_TO_GDPR_PRINCIPLES.get(opp_category, [])


def preprocess_opp115_for_classification(base_path, random_seed=42, threshold="threshold-0.75-overlap-similarity"):
    """
    Main preprocessing function to prepare OPP-115 for GDPR classification.
    
    Args:
        base_path: Path to OPP-115 dataset
        random_seed: Random seed for reproducibility
        threshold: Consolidation threshold to use
    
    Returns:
        train_df, val_df, test_df: DataFrames ready for model training
    """
    print("Loading OPP-115 dataset...")
    
    # Check for consolidation folder (with threshold subdirectories)
    if os.path.exists(os.path.join(base_path, "OPP-115", "consolidation")):
        consolidation_path = os.path.join(base_path, "OPP-115", "consolidation", threshold)
    else:
        consolidation_path = os.path.join(base_path, "consolidation", threshold)
    
    if os.path.exists(consolidation_path):
        print(f"Loading from consolidation folder (threshold: {threshold})...")
        df = load_opp115_consolidation(base_path, threshold)
    else:
        print("Loading from pretty_print folder...")
        df = load_opp115_pretty_print(base_path)
    
    print(f"Loaded {len(df)} segments")
    
    # Clean and preprocess
    if 'text' not in df.columns and 'segment_text' in df.columns:
        df['text'] = df['segment_text']
    
    # Remove empty or very short texts
    df = df[df['text'].notna()]
    df = df[df['text'].str.len() > 10]
    
    # Map categories if present
    if 'category' in df.columns:
        df['gdpr_category'] = df['category'].apply(map_to_gdpr_category)
        df['gdpr_articles'] = df['category'].apply(map_to_gdpr_articles)
        df['gdpr_principles'] = df['category'].apply(map_to_gdpr_principles)
    else:
        # If no category, we need to infer or use a default
        df['gdpr_category'] = 'Other'
        df['gdpr_articles'] = [[]]
        df['gdpr_principles'] = [[]]
    
    # Create label column
    df['label'] = df['gdpr_category'].map(CLASS_LABELS)
    df['target'] = df['gdpr_category']
    
    # Rename text column for compatibility with existing pipeline
    df['Sentence'] = df['text']
    
    # Add ID column
    df['ID'] = range(len(df))
    
    # Split by policy to avoid data leakage
    if 'policy_name' in df.columns:
        policies = df['policy_name'].unique()
        np.random.seed(random_seed)
        np.random.shuffle(policies)
        
        n_policies = len(policies)
        train_idx = int(0.70 * n_policies)
        val_idx = int(0.85 * n_policies)
        
        train_policies = policies[:train_idx]
        val_policies = policies[train_idx:val_idx]
        test_policies = policies[val_idx:]
        
        train_df = df[df['policy_name'].isin(train_policies)].copy()
        val_df = df[df['policy_name'].isin(val_policies)].copy()
        test_df = df[df['policy_name'].isin(test_policies)].copy()
        
        train_df['dataset_type'] = 'train'
        val_df['dataset_type'] = 'val'
        test_df['dataset_type'] = 'test'
    else:
        # Random split if no policy information
        train_df, temp_df = train_test_split(df, test_size=0.30, random_state=random_seed, stratify=df['label'])
        val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=random_seed, stratify=temp_df['label'])
        
        train_df['dataset_type'] = 'train'
        val_df['dataset_type'] = 'val'
        test_df['dataset_type'] = 'test'
    
    print(f"\nDataset splits:")
    print(f"  Train: {len(train_df)} samples")
    print(f"  Validation: {len(val_df)} samples")
    print(f"  Test: {len(test_df)} samples")
    
    print(f"\nClass distribution in training set:")
    print(train_df['target'].value_counts())
    
    return train_df, val_df, test_df


def create_combined_dataset(train_df, val_df, test_df, output_path):
    """
    Create combined dataset files for the pipeline.
    
    Args:
        train_df, val_df, test_df: Split DataFrames
        output_path: Path to save the output files
    """
    # Combine train and val for training file (with dataset_type column)
    train_combined = pd.concat([train_df, val_df], ignore_index=True)
    
    # Select columns compatible with existing pipeline
    columns_to_keep = ['ID', 'Sentence', 'target', 'label', 'dataset_type']
    if 'policy_name' in train_df.columns:
        columns_to_keep.append('policy_name')
    
    train_output = train_combined[columns_to_keep]
    test_output = test_df[columns_to_keep]
    
    # Save to CSV
    train_output.to_csv(os.path.join(output_path, 'opp115_train.csv'), index=False)
    test_output.to_csv(os.path.join(output_path, 'opp115_test.csv'), index=False)
    
    print(f"\nSaved datasets to {output_path}")
    print(f"  opp115_train.csv: {len(train_output)} samples")
    print(f"  opp115_test.csv: {len(test_output)} samples")
    
    return train_output, test_output


def get_class_info():
    """Return class labels and names for reference."""
    return {
        'labels': CLASS_LABELS,
        'names': CLASS_NAMES,
        'gdpr_mapping': SIMPLIFIED_GDPR_MAPPING,
        'articles_mapping': OPP_TO_GDPR_ARTICLES,
        'principles_mapping': OPP_TO_GDPR_PRINCIPLES
    }


if __name__ == "__main__":
    # Test the preprocessing
    import sys
    
    # Base path can be relative or absolute
    if len(sys.argv) > 1:
        base_path = sys.argv[1]
    else:
        # Default: relative to project root
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        base_path = os.path.join(project_root, "Input", "OPP-115")
    
    output_path = os.path.dirname(base_path) if "OPP-115" in base_path else base_path
    
    print(f"Looking for OPP-115 at: {base_path}")
    print(f"Output path: {output_path}")
    
    if os.path.exists(base_path):
        train_df, val_df, test_df = preprocess_opp115_for_classification(base_path)
        create_combined_dataset(train_df, val_df, test_df, output_path)
    else:
        print(f"OPP-115 dataset not found at {base_path}")
        print("Please download from: https://www.usableprivacy.org/static/data/OPP-115_v1_0.zip")
