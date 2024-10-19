import sys
sys.path.append("../")
import urllib.request
import time
import pandas as pd
import os

def get_data(df,image_dir):
    for i, row in df.iterrows():
        url = str(row["image"])
        try:
            time.sleep(0.1)
            urllib.request.urlretrieve(url, f"{image_dir}{i}.jpg")
            row["img_path"] = f"{image_dir}{i}.jpg"
        except:
            row["img_path"] = None
    
    for i, row in df.iterrows():
        img_path = f"{image_dir}{i}.jpg"
        img_path2 = f"{image_dir}{i}.jpg"
        if os.path.exists(img_path2):
            df.at[i, "img_path"] = img_path2  # ファイルが存在する場合はパスを設定
        else:
            df.at[i, "img_path"] = None  # 存在しない場合はNoneを設定
    
    return df

if __name__ == "__main__":
    image_dir = "/root/graduation_thetis/causal-bert-pytorch/input/watch_images/"
    csv_path = "/root/graduation_thetis/causal-bert-pytorch/input/Watches.csv"
    df = pd.read_csv(csv_path)
    df = get_data(df,image_dir)
    df.to_csv("./watch_img.csv",index = None)