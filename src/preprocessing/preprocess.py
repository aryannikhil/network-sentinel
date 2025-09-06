import pandas as pd
import numpy as np

def drop_identifier_columns(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Drop Non-Generalizable Identifiers

    Removes dataset-specific identifiers like IP addresses, ports, and IDs
    that could cause the model to memorize instead of learning patterns.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training dataset.
    test_df : pd.DataFrame
        Test dataset.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Processed train and test datasets without identifier columns.
    """
    drop_cols= ['srcip', 'sport', 'dstip', 'dsport', 'id']
    
    train_df= train_df.drop(columns=drop_cols, errors='ignore')
    test_df= test_df.drop(columns=drop_cols, errors='ignore')

    return train_df, test_df

def add_derived_features(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Add Derived Features
    Adds new behavioral features to both train and test sets:
    - byte_ratio       = sbytes / (dbytes + 1)
    - packet_rate      = (spkts + dpkts) / (dur + 1)
    - bytes_per_pkt    = (sbytes + dbytes) / (spkts + dpkts + 1)
    """

    def _add_features(df: pd.DataFrame) -> pd.DataFrame:
        df= df.copy()
        df['byte_ratio']= df['sbytes'] / (df['dbytes'] + 1)
        df['packet_rate']= (df['spkts'] + df['dpkts']) / (df['dur'] + 1)
        df['bytes_per_pkt']= (df['sbytes'] + df['dbytes']) / (df['spkts'] + df['dpkts'] + 1)
        return df

    return _add_features(train_df), _add_features(test_df)

def add_aggregations_and_flags(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """ 
    Adds:
    - Flow asymmetry ratios (sbytes vs dbytes, spkts vs dpkts)
    - Connection flags derived from 'state' column
    """
    def _add_features(df: pd.DataFrame) -> pd.DataFrame:
        df= df.copy()

        # flow asymetry ratios
        df['sbytes_dbytes_ratio']= df['sbytes'] / (df['dbytes'] + 1)
        df['spkts_dpkts_ratio']= df['spkts'] / (df['dpkts'] + 1)

        # flags from 'state'
        df['is_success']= (df['state'] == 'CON').astype(int)
        df['is_reset']= (df['state'] == 'RST').astype(int)

        return df
        
    return _add_features(train_df), _add_features(test_df)

def transform_features(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
     - Log-transform skewed features (dur, sbytes, dbytes)
    - Bin continuous features (dur into categories) 
    """

    def _transform(df: pd.DataFrame) -> pd.DataFrame:
        df= df.copy()

        # log transform skewed numeric features
        for col in ['dur', 'sbytes', 'dbytes']:
            if col in df.columns:
                df[f'{col}_log']= np.log1p(df[col])

        # binning 'dur' into categories
        if 'dur' in df.columns:
            df['dur_bin']= pd.cut(
                df['dur'],
                bins=[-1, 10, 100, float('inf')],
                labels=['short', 'medium', 'long']
            )

        return df

    return _transform(train_df), _transform(test_df)

def apply_pca(train_df: pd.DataFrame, test_df: pd.DataFrame, n_components: int=5) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply PCA to reduce dimensionality of the dataset.
    Reduces correlated features into principal components.
    NOTE: Tree-based models often don't need PCA. Use mainly for visualization
          or when using linear models.
    """

    numeric_cols= train_df.select_dtypes(include=[np.number]).columns
    # Fit PCA on train numeric features
    pca = PCA(n_components=n_components, random_state=42)
    train_pca = pca.fit_transform(train_df[numeric_cols].fillna(0))
    test_pca = pca.transform(test_df[numeric_cols].fillna(0))

    # Create DataFrames for PCA components
    train_pca_df = pd.DataFrame(train_pca, columns=[f'pca_{i+1}' for i in range(n_components)], index=train_df.index)
    test_pca_df = pd.DataFrame(test_pca, columns=[f'pca_{i+1}' for i in range(n_components)], index=test_df.index)

    # Concatenate with original datasets
    train_df = pd.concat([train_df, train_pca_df], axis=1)
    test_df = pd.concat([test_df, test_pca_df], axis=1)

    return train_df, test_df



def run_feature_engineering(train_path: str, test_path: str, save: bool = True, use_pca: bool = False):
    """
    Runs the full feature engineering pipeline.
    """
    # Load raw datasets
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Sequentially apply feature engineering steps
    train_df, test_df = drop_identifier_columns(train_df, test_df)
    train_df, test_df = add_derived_features(train_df, test_df)
    train_df, test_df = add_aggregations_and_flags(train_df, test_df)
    train_df, test_df = transform_features(train_df, test_df)

    if use_pca:
        train_df, test_df = apply_pca(train_df, test_df, n_components=5)

    # Save final processed datasets
    if save:
        train_df.to_csv("../data/processed/train/train_processed.csv", index=False)
        test_df.to_csv("../data/processed/test/test_processed.csv", index=False)
        print("Engineered features saved to ../data/processed/train/ and ../data/processed/test/")

    return train_df, test_df


if __name__ == "__main__":
    run_feature_engineering(
        train_path="data/raw/training/UNSW_NB15_training-set.csv",
        test_path="data/raw/testing/UNSW_NB15_testing-set.csv",
        save=True,
        use_pca=False   # set to True if you want PCA
    )