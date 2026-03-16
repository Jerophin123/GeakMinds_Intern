import pandas as pd
import numpy as np
import os
import pickle
import sys
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def load_data(file_path):
    print(f"Loading data from {file_path}...")
    try:
        df = pd.read_csv(file_path)
        df.rename(columns={'User ID': 'User_ID'}, inplace=True)
        return df
    except FileNotFoundError:
        print(f"Error: Could not find file at {file_path}")
        sys.exit(1)

def transform_features(df, data_dir):
    print("Transforming features...")
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df = df.sort_values(by=['User_ID', 'Timestamp'])
    
    if df['Conversion'].dtype == 'object':
        df['Conversion'] = df['Conversion'].map({'Yes': 1, 'No': 0})
        
    def aggregate_user(user_df):
        user_df = user_df.sort_values('Timestamp')
        first_touch = user_df.iloc[0]
        last_touch = user_df.iloc[-1]
        time_to_conversion = (last_touch['Timestamp'] - first_touch['Timestamp']).total_seconds() / 3600.0
        
        return pd.Series({
            'User_ID': first_touch['User_ID'],
            'num_touchpoints': len(user_df),
            'first_channel': first_touch['Channel'],
            'last_channel': last_touch['Channel'],
            'unique_channels': user_df['Channel'].nunique(),
            'time_to_conversion_hours': time_to_conversion,
            'converted': last_touch['Conversion']
        })
        
    user_features = df.groupby('User_ID').apply(aggregate_user).reset_index(drop=True)
    user_features_encoded = pd.get_dummies(user_features, columns=['first_channel', 'last_channel'])
    
    # Save processed data
    os.makedirs(data_dir, exist_ok=True)
    processed_data_path = os.path.join(data_dir, 'processed_data.csv')
    user_features_encoded.to_csv(processed_data_path, index=False)
    print(f"Saved processed features to {processed_data_path}")
    
    return user_features_encoded

def train_model(df):
    print("Training model...")
    X = df.drop(columns=['User_ID', 'converted'])
    y = df['converted']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Best parameters from GridSearch in notebook (example)
    model = XGBClassifier(
        n_estimators=100, 
        max_depth=3, 
        learning_rate=0.1, 
        eval_metric='logloss', 
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    print("\n--- Model Evaluation ---")
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
    print(f"F1 Score:  {f1_score(y_test, y_pred):.4f}")
    print(f"ROC-AUC:   {roc_auc_score(y_test, y_prob):.4f}\n")
    
    return model, X_train.columns

def save_model(model, output_dir='model'):
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, 'trained_model.pkl')
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
        
    print(f"Model successfully saved to {model_path}")

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_dir = os.path.join(project_root, 'data')
    raw_data_path = os.path.join(data_dir, 'raw_data.csv')
    
    df_raw = load_data(raw_data_path)
    df_processed = transform_features(df_raw, data_dir)
    
    model, features = train_model(df_processed)
    
    model_dir = os.path.join(project_root, 'model')
    save_model(model, model_dir)
    
    print("\nPipeline completed successfully!")

if __name__ == "__main__":
    main()
