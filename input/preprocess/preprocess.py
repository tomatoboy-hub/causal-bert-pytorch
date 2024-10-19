import sys
sys.path.append("../../")
sys.path.append("../../../")

import os
from PIL import Image
from torchvision import transforms
import pandas as pd
import torch
import timm
import pickle
import numpy as np

def preprocessing(input_df):
    """
    欠損値や文字の処理を行う
    """
    def price2yen(input_df):
        output = input_df.copy()
        output = output.dropna(subset=["actual_price"])
        output["actual_price"] = output["actual_price"].str.strip("₹")
        output["actual_price"] = output["actual_price"].str.replace(",","").astype(float)
        output["actual_price_yen"] = output["actual_price"] * 110
        return output
    
    def no_of_rate(input_df):
        output = input_df.copy()
        output = output.dropna(subset=["no_of_ratings"])
        #文字情報が含まれている場合、エラーが発生する可能性があるため、エラーハンドリングを追加
        output["no_of_ratings"] = pd.to_numeric(output["no_of_ratings"].str.replace(",",""), errors='coerce').fillna(0).astype(int)
        return output
    
    def img_path(input_df):
        output = input_df.copy()
        output = output.dropna(subset=["img_path"])
        return output
    
    output = price2yen(input_df)
    output = no_of_rate(output)
    output = img_path(output)
    return output

def filter_outlier(output_df):
    # 四分位範囲（IQR）の計算
    Q1 = output_df["actual_price_yen"].quantile(0.25)
    Q3 = output_df["actual_price_yen"].quantile(0.75)
    IQR = Q3 - Q1

    # 外れ値の閾値を設定
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # 外れ値を除去
    filtered_df = output_df[(output_df["actual_price_yen"] >= lower_bound) & (output_df["actual_price_yen"] <= upper_bound)]
    return filtered_df

def make_treatment(filtered_df):
    # "actual_price_yen"の平均を計算
    mean_price = filtered_df["actual_price_yen"].mean()
    # "price_ave"列を追加
    filtered_df["price_ave"] = filtered_df["actual_price_yen"].apply(lambda x: 1 if x > mean_price else 0)
    return filtered_df

def make_confounder(df):
    def is_dark_or_light(image_path, threshold=160):
        # 画像を読み込んでRGBに変換
        img = Image.open(image_path).convert('RGB')
        # 画像をNumPy配列に変換
        img_np = np.array(img)
        # 輝度の計算 (R, G, B の加重平均)
        brightness = 0.299 * img_np[:,:,0] + 0.587 * img_np[:,:,1] + 0.114 * img_np[:,:,2]
        # 画像全体の平均輝度を計算
        avg_brightness = np.mean(brightness)
        print(f"平均輝度: {avg_brightness}")
        # 平均輝度が閾値より低ければ「黒っぽい」、高ければ「白っぽい」
        if avg_brightness < threshold:
            print("画像は黒っぽいです。")
            return "dark"
        else:
            print("画像は白っぽいです。")
            return "light"
    
    df["light_or_dark"] = df["img_path"].apply(is_dark_or_light)    
    df["light_or_dark"] = df["light_or_dark"].apply(lambda x : 1 if x == "light" else 0)
    return df

if __name__ == "__main__":
    csv_path = "/root/graduation_thetis/causal-bert-pytorch/input/watch_img.csv"
    df = pd.read_csv(csv_path)
    df = preprocessing(df)
    df = filter_outlier(df)
    df = make_treatment(df)
    df = make_confounder(df)
    df.to_csv("root/graduation_thetis/causal-bert-pytorch/input/watch_train.csv",index = None)
