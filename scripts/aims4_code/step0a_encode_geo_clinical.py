"""
Step 0A: Encode GEO Clinical Variables for Late Fusion (Chapter 3, Aim 1)

Purpose:
    Reads raw clinical CSVs for GSE42743 and GSE41613, encodes variables
    into the format needed for ridge Cox clinical risk score models.

Encoding Specification:
    GSE42743 (n=71):
        - gender: already 0/1 (keep as-is)
        - age_zscore: continuous, z-scored across 71 patients
        - smoking: Ever=1, Never=0
        - T_stage: T1-T2=0, T3-T4=1
        - N_stage: N0=0, N+=1

    GSE41613 (n=97):
        - gender: already 0/1 (keep as-is)
        - tumor_stage: I/II=0, III/IV=1
        - age_binary: <60=0 (bins 40-49, 50-59), >=60=1 (bin 60-88)
        Note: age cutpoint differs from TCGA/ORIEN (<=65/>65) because
        GEO provides binned age only. Acknowledged as limitation.

    Missing values: encoded as separate indicator category.
    (No missingness confirmed in GSE42743; GSE41613 to be verified.)

Input:
    data/external/GSE42743_clinical.csv
    data/external/GSE41613_clinical.csv

Output:
    data/external/GSE42743_clinical_encoded.csv
    data/external/GSE41613_clinical_encoded.csv

Author: Phuong Nguyen
Date: March 2026
"""

import pandas as pd
import numpy as np
import sys
import os

# ============================================================================
# Configuration
# ============================================================================

INPUT_DIR = "data/external"
OUTPUT_DIR = "data/external"

GSE42743_INPUT = os.path.join(INPUT_DIR, "GSE42743_clinical.csv")
GSE41613_INPUT = os.path.join(INPUT_DIR, "GSE41613_clinical.csv")

GSE42743_OUTPUT = os.path.join(OUTPUT_DIR, "GSE42743_clinical_encoded.csv")
GSE41613_OUTPUT = os.path.join(OUTPUT_DIR, "GSE41613_clinical_encoded.csv")

# ============================================================================
# GSE42743 Encoding
# ============================================================================

def encode_gse42743(filepath):
    """
    Encode GSE42743 clinical variables.

    Predictors for clinical model (5):
        gender, age_zscore, smoking, T_stage, N_stage
    """
    df = pd.read_csv(filepath)

    print("=" * 60)
    print("GSE42743: Raw Data Summary")
    print("=" * 60)
    print(f"  Samples: {len(df)}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Missing values per column:\n{df.isnull().sum()}")
    print()

    encoded = pd.DataFrame()
    encoded["sampleID"] = df["sampleID"]

    # Gender: already 0/1 — verify and keep
    assert df["gender"].isin([0, 1]).all(), "Unexpected gender values"
    encoded["gender"] = df["gender"].astype(int)

    # Age: continuous z-score
    assert df["age"].notnull().all(), "Missing age values found"
    age_mean = df["age"].mean()
    age_std = df["age"].std(ddof=1)  # sample std
    encoded["age_zscore"] = (df["age"] - age_mean) / age_std

    # Smoking: Ever=1, Never=0
    smoking_map = {"Ever": 1, "Never": 0}
    unmapped_smoking = set(df["smoking"].unique()) - set(smoking_map.keys())
    if unmapped_smoking:
        print(f"  WARNING: Unmapped smoking values: {unmapped_smoking}")
        print("  These will be encoded as 'Unknown' indicator variable")
        # Create Unknown indicator if needed
        encoded["smoking"] = df["smoking"].map(smoking_map)
        encoded["smoking_unknown"] = encoded["smoking"].isnull().astype(int)
        encoded["smoking"] = encoded["smoking"].fillna(0)
    else:
        encoded["smoking"] = df["smoking"].map(smoking_map).astype(int)

    # T_stage: T1-T2=0, T3-T4=1
    tstage_map = {"T1-T2": 0, "T3-T4": 1}
    unmapped_tstage = set(df["T_stage"].unique()) - set(tstage_map.keys())
    if unmapped_tstage:
        print(f"  WARNING: Unmapped T_stage values: {unmapped_tstage}")
        encoded["T_stage"] = df["T_stage"].map(tstage_map)
        encoded["T_stage_unknown"] = encoded["T_stage"].isnull().astype(int)
        encoded["T_stage"] = encoded["T_stage"].fillna(0)
    else:
        encoded["T_stage"] = df["T_stage"].map(tstage_map).astype(int)

    # N_stage: N0=0, N+=1
    nstage_map = {"N0": 0, "N+": 1}
    unmapped_nstage = set(df["N_stage"].unique()) - set(nstage_map.keys())
    if unmapped_nstage:
        print(f"  WARNING: Unmapped N_stage values: {unmapped_nstage}")
        encoded["N_stage"] = df["N_stage"].map(nstage_map)
        encoded["N_stage_unknown"] = encoded["N_stage"].isnull().astype(int)
        encoded["N_stage"] = encoded["N_stage"].fillna(0)
    else:
        encoded["N_stage"] = df["N_stage"].map(nstage_map).astype(int)

    # --- Verification ---
    print("GSE42743: Encoded Summary")
    print("-" * 40)
    print(f"  Samples: {len(encoded)}")
    print(f"  Columns: {list(encoded.columns)}")
    print(f"  Missing after encoding: {encoded.isnull().sum().sum()}")
    print()
    print("  gender distribution:")
    print(f"    {encoded['gender'].value_counts().to_dict()}")
    print()
    print(f"  age_zscore: mean={encoded['age_zscore'].mean():.4f}, "
          f"std={encoded['age_zscore'].std():.4f}")
    print(f"  age raw: mean={age_mean:.1f}, std={age_std:.1f}, "
          f"min={df['age'].min()}, max={df['age'].max()}")
    print()
    print("  smoking distribution:")
    print(f"    {encoded['smoking'].value_counts().to_dict()}")
    print()
    print("  T_stage distribution:")
    print(f"    {encoded['T_stage'].value_counts().to_dict()}")
    print()
    print("  N_stage distribution:")
    print(f"    {encoded['N_stage'].value_counts().to_dict()}")
    print()

    return encoded


# ============================================================================
# GSE41613 Encoding
# ============================================================================

def encode_gse41613(filepath):
    """
    Encode GSE41613 clinical variables.

    Predictors for clinical model (3):
        gender, tumor_stage, age_binary

    Note: age cutpoint is <60 vs >=60 (based on available bins),
    which differs from TCGA/ORIEN <=65/>65. Acknowledged as limitation.
    """
    df = pd.read_csv(filepath)

    print("=" * 60)
    print("GSE41613: Raw Data Summary")
    print("=" * 60)
    print(f"  Samples: {len(df)}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Missing values per column:\n{df.isnull().sum()}")
    print()

    encoded = pd.DataFrame()
    encoded["sampleID"] = df["sampleID"]

    # Gender: already 0/1 — verify and keep
    assert df["gender"].isin([0, 1]).all(), "Unexpected gender values"
    encoded["gender"] = df["gender"].astype(int)

    # Tumor stage: I/II=0, III/IV=1
    stage_map = {"I/II": 0, "III/IV": 1}
    unmapped_stage = set(df["tumor_stage"].unique()) - set(stage_map.keys())
    if unmapped_stage:
        print(f"  WARNING: Unmapped tumor_stage values: {unmapped_stage}")
        encoded["tumor_stage"] = df["tumor_stage"].map(stage_map)
        encoded["tumor_stage_unknown"] = encoded["tumor_stage"].isnull().astype(int)
        encoded["tumor_stage"] = encoded["tumor_stage"].fillna(0)
    else:
        encoded["tumor_stage"] = df["tumor_stage"].map(stage_map).astype(int)

    # Age: binary <60 vs >=60
    # Bins: 40-49 -> 0, 50-59 -> 0, 60-88 -> 1
    age_map = {"19-39": 0, "40-49": 0, "50-59": 0, "60-88": 1}
    unmapped_age = set(df["age_bin"].unique()) - set(age_map.keys())
    if unmapped_age:
        print(f"  WARNING: Unmapped age_bin values: {unmapped_age}")
        encoded["age_binary"] = df["age_bin"].map(age_map)
        encoded["age_unknown"] = encoded["age_binary"].isnull().astype(int)
        encoded["age_binary"] = encoded["age_binary"].fillna(0)
    else:
        encoded["age_binary"] = df["age_bin"].map(age_map).astype(int)

    # --- Verification ---
    print("GSE41613: Encoded Summary")
    print("-" * 40)
    print(f"  Samples: {len(encoded)}")
    print(f"  Columns: {list(encoded.columns)}")
    print(f"  Missing after encoding: {encoded.isnull().sum().sum()}")
    print()
    print("  gender distribution:")
    print(f"    {encoded['gender'].value_counts().to_dict()}")
    print()
    print("  tumor_stage distribution:")
    print(f"    {encoded['tumor_stage'].value_counts().to_dict()}")
    print()
    print("  age_binary distribution (<60=0, >=60=1):")
    print(f"    {encoded['age_binary'].value_counts().to_dict()}")
    print()

    return encoded


# ============================================================================
# Main
# ============================================================================

def main():
    # --- Verify input files exist ---
    for fpath in [GSE42743_INPUT, GSE41613_INPUT]:
        if not os.path.exists(fpath):
            print(f"ERROR: Input file not found: {fpath}")
            sys.exit(1)

    # --- Encode GSE42743 ---
    gse42743_encoded = encode_gse42743(GSE42743_INPUT)
    gse42743_encoded.to_csv(GSE42743_OUTPUT, index=False)
    print(f"  Saved: {GSE42743_OUTPUT}")
    print()

    # --- Encode GSE41613 ---
    gse41613_encoded = encode_gse41613(GSE41613_INPUT)
    gse41613_encoded.to_csv(GSE41613_OUTPUT, index=False)
    print(f"  Saved: {GSE41613_OUTPUT}")
    print()

    # --- Final summary ---
    print("=" * 60)
    print("ENCODING COMPLETE")
    print("=" * 60)
    print()
    print("Clinical model variables per dataset:")
    print("  GSE42743 (5): gender, age_zscore, smoking, T_stage, N_stage")
    print("  GSE41613 (3): gender, tumor_stage, age_binary")
    print()
    print("Next step: Step 0B — generate per-patient out-of-fold")
    print("gene risk scores using 64 consensus genes.")


if __name__ == "__main__":
    main()
