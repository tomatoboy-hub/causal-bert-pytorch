import pandas as pd
import numpy as np
import numpy.random as random
from scipy.special import expit


def calculate_propensity_score(df, confounder, treatment):
    confounder_df = df[df[confounder] == 0]
    probability_0 = (confounder_df[treatment] == 1).mean()
    print(f"{confounder}が0のもののうち{treatment}が1である確率:", probability_0)
    confounder_df = df[df[confounder] == 1]
    probability_1 = (confounder_df[treatment] == 1).mean()
    print(f"{confounder}が1のもののうち{treatment}が1である確率:", probability_1)
    return probability_0,probability_1

def outcome_sim(beta0, beta1, gamma, treatment, confounding, noise, setting = "simple"):
    if setting == "simple":
        y0 = beta1 * confounding
        y1 = beta0 * y0

        simulated_score = (1. - treatment) * y0 + treatment * y1 + gamma * noise
    return simulated_score, y0,y1

def make_price_dark_probs(df,treat_strength,con_strength,probability_0, probability_1,noise_level, setting = "simple", seed = 0):
    price_ave_given_dark_probs = np.array([probability_0,probability_1])

    np.random.seed(seed)
    all_noise = np.array(random.normal(0,1,len(df)), dtype = np.float32)
    all_thresholds = np.array(random.uniform(0,1,len(df)), dtype = np.float32)
    
    outcomes = []
    y0s = []
    y1s = []

    for i, data in df.iterrows():
        light_or_dark = data["light_or_dark"]
        treatment = data["price_ave"]

        confounding = 3.0 * (price_ave_given_dark_probs[light_or_dark] - 0.25)
        noise = all_noise[i]
        y,y0,y1 = outcome_sim(treat_strength, con_strength, noise_level,treatment, confounding, noise, setting = setting)
        simulated_prob = expit(y)
        y0 = expit(y0)
        y1 = expit(y1)
        threshold = all_thresholds[i]
        simulated_outcome = 1 if simulated_prob > threshold else 0
        outcomes.append(simulated_outcome)
        y0s.append(y0)
        y1s.append(y1)
        
    df['outcome'] = outcomes
    df['y0'] = y0s
    df['y1'] = y1s
    return df

if __name__ == "__main__":
    csv_path = "root/graduation_thetis/causal-bert-pytorch/input/watch_train.csv"
    confounder = "light_or_dark"
    treatment = "price_ave"
    df = pd.read_csv(csv_path)
    probability_0, probability_1 = calculate_propensity_score(df, confounder, treatment)
    df = make_price_dark_probs(df,0.5, 5.0,probability_0,probability_1,0.0,"simple", 0)
    df.to_csv("root/graduation_thetis/causal-bert-pytorch/input/watch_output.csv", index = None)
