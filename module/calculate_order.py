import pandas as pd

def calculate_goodness(df, n, evaluater_num, remove_top_n):
    
    result_matrix = pd.DataFrame(
        index=[i for i in range(remove_top_n,n+remove_top_n)], 
        columns=[i for i in range(remove_top_n,n+remove_top_n)]
    )

    result_matrix = result_matrix.fillna(0)

    for i in range(len(df)):

        if df.iloc[i,:]["A_index"] < remove_top_n or df.iloc[i,:]["B_index"] < remove_top_n:
            continue
        
        if df.iloc[i,:]["A_index"] == df.iloc[i,:]["B_index"]:
            continue

        elif df.iloc[i,:]["result"] == "A":
            result_matrix.loc[df.iloc[i,:]["A_index"], df.iloc[i,:]["B_index"]] += 1
        elif df.iloc[i,:]["result"] == "B":
            result_matrix.loc[df.iloc[i,:]["A_index"], df.iloc[i,:]["B_index"]] -= 1
        else:
            raise ValueError("Invalid result")

    X_i = result_matrix.sum(axis=1)
    X_j = result_matrix.sum(axis=0)

    a = (X_i - X_j) / (2 * n * evaluater_num)

    return a