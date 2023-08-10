import pandas as pd 


data = pd.read_csv("./data_10k.csv", na_filter=False)

df = pd.DataFrame(columns=['instruction', 'input', 'output', 'text'])

k = 0
for i in range(len(data)):
    if not data['text'][i]: continue
    raw_data = data['text'][i].split("\n\n")
    instruction_, input_, output_, text_ = '', '', '', ''
    for line in raw_data:
        if "### Instruction" in line:
            instruction_ = line.split("\n")[1]
        if "### Input:" in line:
            input_ = line.split("\n")[1]
        if "### Response" in line:
            output_ = line.split("\n")[1]
    text_ = data['text'][i]
    df.loc[k] = [instruction_, input_, output_, text_]
    k += 1

df.to_csv('data_10k_processed.csv', index=False)
