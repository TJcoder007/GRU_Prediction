## -*- coding: utf-8 -*-
import pandas as pd

theta = 0
def map_func(x,theta):
        if x > theta:
            return 1
        else:
            return -1

predict_df = pd.read_csv(r"C:\Users\Jiwei Tu\Desktop\predict_df.csv",encoding="utf_8_sig")
predicted_stock_price = predict_df.applymap(lambda x: map_func(x,theta))
predicted_stock_price.to_csv(r"C:\Users\Jiwei Tu\Desktop\predict_df_01.csv",encoding="utf_8_sig")
real_df = pd.to_csv(r"C:\Users\Jiwei Tu\Desktop\real_df.csv",encoding="utf_8_sig")
real_stock_price = real_df.applymap(lambda x: map_func(x,theta))
real_stock_price.to_csv(r"C:\Users\Jiwei Tu\Desktop\real_df_01.csv",encoding="utf_8_sig")