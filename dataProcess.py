import pandas as pd

member_df = pd.read_csv(r"C:\Users\Jiwei Tu\Desktop\深度学习预测\中证500\中证500\zz500_member - 副本.csv")
close_df = pd.read_csv(r"C:\Users\Jiwei Tu\Desktop\深度学习预测\中证500\中证500\real_close.csv")
high_df = pd.read_csv(r"C:\Users\Jiwei Tu\Desktop\深度学习预测\中证500\中证500\high_p.csv")
low_df = pd.read_csv(r"C:\Users\Jiwei Tu\Desktop\深度学习预测\中证500\中证500\low_p.csv")
pct_df = pd.read_csv(r"C:\Users\Jiwei Tu\Desktop\深度学习预测\中证500\中证500\handledData\pct.csv")
volume_df = pd.read_csv(r"C:\Users\Jiwei Tu\Desktop\深度学习预测\中证500\中证500\volume_p.csv")


column_names =[]
#print(member_df)
for i in range(1,len(member_df.columns)):
    if(member_df.iloc[750,i] == 730):
        column_names.append(member_df.columns[i])

T_close = pct_df[['TRADE_DT']]
for j in range(len(column_names)):
    for k in range(len(close_df)):
        if(close_df.columns[k] == column_names[j]):
            T_close = pd.concat([T_close,close_df[column_names[j]]],axis=1) 
T_close = T_close.loc[2644:3408,:]          
T_close.to_csv(r"C:\Users\Jiwei Tu\Desktop\深度学习预测\中证500\中证500\handledData\T_close.csv",encoding="utf_8_sig")

T_high = pct_df[['TRADE_DT']]
for j in range(len(column_names)):
    for k in range(len(high_df)):
        if(high_df.columns[k] == column_names[j]):
            T_high = pd.concat([T_high,high_df[column_names[j]]],axis=1) 
T_high = T_high.loc[2644:3408,:]          
T_high.to_csv(r"C:\Users\Jiwei Tu\Desktop\深度学习预测\中证500\中证500\handledData\T_high.csv",encoding="utf_8_sig")

T_low = pct_df[['TRADE_DT']]
for j in range(len(column_names)):
    for k in range(len(low_df)):
        if(low_df.columns[k] == column_names[j]):
            T_low = pd.concat([T_low,low_df[column_names[j]]],axis=1) 
T_low = T_low.loc[2644:3408,:]          
T_low.to_csv(r"C:\Users\Jiwei Tu\Desktop\深度学习预测\中证500\中证500\handledData\T_low.csv",encoding="utf_8_sig")

T_pct = pct_df[['TRADE_DT']]
for j in range(len(column_names)):
    for k in range(len(pct_df)):
        if(pct_df.columns[k] == column_names[j]):
            T_pct = pd.concat([T_pct,pct_df[column_names[j]]],axis=1) 
T_pct = T_pct.loc[2644:3408,:]          
T_pct.to_csv(r"C:\Users\Jiwei Tu\Desktop\深度学习预测\中证500\中证500\handledData\T_pct.csv",encoding="utf_8_sig")

T_volume = pct_df[['TRADE_DT']]
for j in range(len(column_names)):
    for k in range(len(volume_df)):
        if(volume_df.columns[k] == column_names[j]):
            T_volume = pd.concat([T_volume,volume_df[column_names[j]]],axis=1) 
T_volume = T_volume.loc[2644:3408,:]          
T_volume.to_csv(r"C:\Users\Jiwei Tu\Desktop\深度学习预测\中证500\中证500\handledData\T_volume.csv",encoding="utf_8_sig")