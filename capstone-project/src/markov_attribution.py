import pandas as pd
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import shap

def prepare_journeys(df):
    """
    Constructs user journeys as ordered sequences of channels.
    Adds 'Start' and 'Conversion' (or 'Null') states to each journey.
    """
    print("Preparing user journeys...")
    # Ensure Timestamp is datetime
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df_sorted = df.sort_values(by=['User_ID', 'Timestamp']).copy()
    
    journeys = []
    
    # Map conversion properly
    if df_sorted['Conversion'].dtype == 'object':
        df_sorted['numeric_conv'] = df_sorted['Conversion'].map({'Yes': 1, 'No': 0})
    else:
        df_sorted['numeric_conv'] = df_sorted['Conversion']
        
    for user_id, user_data in df_sorted.groupby('User_ID'):
        channels = user_data['Channel'].tolist()
        converted = user_data['numeric_conv'].iloc[-1]
        
        path = ['Start'] + channels + (['Conversion'] if converted == 1 else ['Null'])
        journeys.append(path)
        
    return journeys

def build_transition_matrix(journeys):
    """
    Builds a transition probability matrix between all states.
    Normalizes transitions to probabilities.
    """
    transitions = {}
    states = set()
    
    for path in journeys:
        for i in range(len(path) - 1):
            source = path[i]
            target = path[i+1]
            states.add(source)
            states.add(target)
            
            if source not in transitions:
                transitions[source] = {}
            if target not in transitions[source]:
                transitions[source][target] = 0
                
            transitions[source][target] += 1
            
    prob_matrix = {}
    for source, targets in transitions.items():
        prob_matrix[source] = {}
        total = sum(targets.values())
        for target, count in targets.items():
            prob_matrix[source][target] = count / total
            
    # Add absorbing states if they aren't fully populated
    if 'Conversion' not in prob_matrix:
        prob_matrix['Conversion'] = {'Conversion': 1.0}
    else:
        prob_matrix['Conversion'] = {'Conversion': 1.0}
        
    if 'Null' not in prob_matrix:
        prob_matrix['Null'] = {'Null': 1.0}
    else:
        prob_matrix['Null'] = {'Null': 1.0}
        
    states.add('Conversion')
    states.add('Null')
    
    states_list = list(states)
    df_trans = pd.DataFrame(0.0, index=states_list, columns=states_list)
    
    for source in prob_matrix:
        for target in prob_matrix[source]:
            df_trans.loc[source, target] = prob_matrix[source][target]
            
    return df_trans, states_list

def calculate_conversion_prob(df_trans, max_steps=50):
    """
    Compute probability of reaching 'Conversion' from 'Start'
    """
    if 'Start' not in df_trans.index or 'Conversion' not in df_trans.columns:
        return 0.0
        
    n_states = len(df_trans.index)
    v = np.zeros(n_states)
    start_idx = df_trans.index.get_loc('Start')
    v[start_idx] = 1.0
    
    M = df_trans.values
    for _ in range(max_steps):
        v = v.dot(M)
        
    conv_idx = df_trans.columns.get_loc('Conversion')
    return v[conv_idx]

def calculate_markov_attribution(journeys):
    """
    Calculates channel contribution using the removal effect.
    Returns channel attribution scores as a dictionary.
    """
    print("Calculating Markov attribution scores via removal effect...")
    df_trans, states = build_transition_matrix(journeys)
    baseline_conv = calculate_conversion_prob(df_trans)
    
    channels = [s for s in states if s not in ['Start', 'Conversion', 'Null']]
    removal_effects = {}
    
    for channel in channels:
        # Create a modified transition matrix
        df_mod = df_trans.copy()
        
        # When channel is removed, all transitions going to it go to 'Null'
        for source in df_mod.index:
            if source != channel and df_mod.loc[source, channel] > 0:
                df_mod.loc[source, 'Null'] += df_mod.loc[source, channel]
                df_mod.loc[source, channel] = 0.0
                
        mod_conv = calculate_conversion_prob(df_mod)
        removal_effect = (baseline_conv - mod_conv) / baseline_conv if baseline_conv > 0 else 0
        removal_effects[channel] = removal_effect
        
    # Normalize contributions
    total_effect = sum(removal_effects.values())
    markov_attribution = {}
    if total_effect > 0:
        for ch, eff in removal_effects.items():
            markov_attribution[ch] = round(float(eff / total_effect), 4)
    else:
        markov_attribution = {ch: 0.0 for ch in channels}
        
    # Sort dict by value descending
    markov_attribution_sorted = dict(sorted(markov_attribution.items(), key=lambda item: item[1], reverse=True))
    
    return markov_attribution_sorted

def calculate_shap_importance(model, X_train):
    """
    Computes SHAP values and aggregates them by channel to compare with Markov scores.
    """
    print("Calculating SHAP-based feature importance...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)
    
    # XGBoost can return 2D or 3D shap_values depending on objective
    if len(np.shape(shap_values)) == 3:
        shap_values_abs = np.abs(shap_values[:, :, 1]).mean(axis=0)
    else:
        shap_values_abs = np.abs(shap_values).mean(axis=0)
        
    feature_importance = dict(zip(X_train.columns, shap_values_abs))
    
    # Aggregate SHAP to channel level using dummy variables conventions
    channel_shap = {}
    for col, imp in feature_importance.items():
        ch = None
        if col.startswith('first_channel_'):
            ch = col.replace('first_channel_', '')
        elif col.startswith('last_channel_'):
            ch = col.replace('last_channel_', '')
            
        if ch:
            if ch not in channel_shap:
                channel_shap[ch] = 0.0
            channel_shap[ch] += imp
            
    # Normalize SHAP
    total_shap = sum(channel_shap.values())
    shap_normalized = {}
    if total_shap > 0:
        for ch, val in channel_shap.items():
            shap_normalized[ch] = round(float(val / total_shap), 4)
    else:
        shap_normalized = {ch: 0.0 for ch in channel_shap.keys()}
        
    # Sort dict by value descending
    shap_normalized_sorted = dict(sorted(shap_normalized.items(), key=lambda item: item[1], reverse=True))
    
    return shap_normalized_sorted, feature_importance

def plot_attributions(markov_scores, shap_scores, output_path):
    """
    Generates a bar chart comparing SHAP vs Markov attribution.
    """
    print("Generating comparison visualization...")
    all_channels = sorted(list(set(list(markov_scores.keys()) + list(shap_scores.keys()))))
    
    m_vals = [markov_scores.get(c, 0.0) for c in all_channels]
    s_vals = [shap_scores.get(c, 0.0) for c in all_channels]
    
    x = np.arange(len(all_channels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, m_vals, width, label='Markov Effect', color='#1f77b4')
    rects2 = ax.bar(x + width/2, s_vals, width, label='SHAP Importance', color='#ff7f0e')
    
    ax.set_ylabel('Normalized Score')
    ax.set_title('Channel Attribution: Markov vs SHAP')
    ax.set_xticks(x)
    ax.set_xticklabels(all_channels, rotation=45, ha='right')
    ax.legend()
    
    fig.tight_layout()
    plt.savefig(output_path)
    plt.close()

def generate_comparison_summary(markov_scores, shap_scores):
    """
    Creates a text summary contrasting the two methods.
    """
    if not markov_scores or not shap_scores:
        return "Not enough data to summarize differences."
        
    top_markov = list(markov_scores.keys())[0]
    top_shap = list(shap_scores.keys())[0]
    
    summary = f"Comparison Summary:\n"
    summary += "- Predictive Importance (SHAP): Focuses heavily on channels primarily acting as immediate conversion predictors across first/last touches.\n"
    summary += "- Sequential Contribution (Markov): Highlights holistic channel value tracking overall interconnected pathways.\n\n"
    
    if top_markov == top_shap:
        summary += f"Both methodologies identify '{top_markov}' as the single most critical channel.\n"
    else:
        summary += f"Methodologies diverge on the top channel: SHAP prioritizes '{top_shap}' while Markov highlights '{top_markov}' sequences.\n"
        
    return summary

def calculate_and_compare_attributions(df_raw, df_features, model, data_dir):
    """
    Main orchestrator for calculating Markov constraints and comparing with SHAP.
    """
    # 1. Calculate Markov Attribution
    journeys = prepare_journeys(df_raw)
    markov_attribution = calculate_markov_attribution(journeys)
    
    # 2. Calculate SHAP
    X_train = df_features.drop(columns=['User_ID', 'converted'])
    shap_importance, _ = calculate_shap_importance(model, X_train)
    
    # 3. Create Summary
    summary = generate_comparison_summary(markov_attribution, shap_importance)
    
    # 4. Compile final data JSON dict
    output_dict = {
        "markov_attribution": markov_attribution,
        "shap_importance": shap_importance,
        "comparison_summary": summary
    }
    
    # Save raw dict
    json_path = os.path.join(data_dir, 'attribution_comparison.json')
    with open(json_path, 'w') as f:
        json.dump(output_dict, f, indent=4)
        
    # Save markov as csv explicitly as requested
    csv_path = os.path.join(data_dir, 'markov_attribution.csv')
    pd.DataFrame(list(markov_attribution.items()), columns=['Channel', 'Markov_Score']).to_csv(csv_path, index=False)
    
    # Visualize
    plot_path = os.path.join(data_dir, 'attribution_comparison.png')
    plot_attributions(markov_attribution, shap_importance, plot_path)
    
    print(f"Attribution analysis saved to {data_dir}.")
    print(f"-> Dictionary   : {json_path}")
    print(f"-> CSV          : {csv_path}")
    print(f"-> Visualization: {plot_path}")
    print(summary)
    
    return output_dict
