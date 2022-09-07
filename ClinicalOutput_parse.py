import pandas as pd
from pathlib import Path
"""
This is the script used to parse the clinical output into the data structure per Qin's request
"""
file = r'C:\Users\Public\Documents\SynologyDrive\Projects\LAMP-Covid Sensor\EUA_Studies\Clinical_Study\ResultArchive\ClinicalOutput_raw2.csv' #This is the file location of the compiled .csv
df = pd.read_csv(file)

savepath = fr'{Path(file).absolute().parent}\ClinicalOutput_parsed2.csv'
with open(savepath, 'w') as f:
    f.write('"id", "sample type", "reader id", "date", "c1 result", "c1 ct", "c1 pr", "c1 sd", "c4 result", "c4 ct", "c4 pr", "c4 sd", "led result"\n')

# df.iloc[1]

result = ''
for i in range(len(df)):
    print(i)
    sample = df.iloc[i]
    sample_type, id, ch = sample['Name'].split('-')
    reader_id = sample['Device'].split('-')[1]
    ct = sample['hyperCt']
    pr = sample['Pr']
    sd = sample['Sd5m']
    date = sample['Date']
    calling = sample['Predict']

    if ch == 'C1':
        result += '+' if sample['Predict'] == 'Positive' else '-'
        msg = f'{id}, {sample_type}, {reader_id}, {date}, {calling}, {ct}, {pr}, {sd},'
    elif ch == 'C4':
        result += '+' if sample['Predict'] == 'Positive' else '-'
        if result == '+-':
            led_result = 'Positive'
        elif result == '++':
            led_result = 'Positive'
        elif result == '-+':
            led_result = 'Negative'
        elif result == '--':
            led_result = 'Invalid'
        msg = f'{calling}, {ct}, {pr}, {sd}, {led_result}\n'
    with open(savepath, 'a') as f:
        f.write(msg)
