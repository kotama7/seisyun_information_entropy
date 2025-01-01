import pandas as pd
import argparse
import os

from module.seisyun import seisyun
from module.set_seed import set_seed
from module.paired_comparison import paired_comparison

def main(args):

    set_seed(args.seed)

    if not os.path.exists(args.sentence_output):

        seisyun_dict = {}

        while len(seisyun_dict) < args.sentence_num:
            score, sentence = seisyun()
            seisyun_dict[sentence] = score

        df = pd.DataFrame(seisyun_dict.items(), columns=["sentence", "score"]).sort_values("score", ascending=False)

        df.to_csv(args.sentence_output, index=False)
    
    else:
    
        df = pd.read_csv(args.sentence_output, index_col=False)

    # Use only the top N sentences
    df = df.head(args.use_sentence_num)

    comparison_df = paired_comparison(df)

    comparison_df.to_csv(args.comparison_output, index=False)


if __name__ == "__main__":
    
    argparser = argparse.ArgumentParser()

    argparser.add_argument("--sentence_num", type=int, default=1000)
    argparser.add_argument("--use_sentence_num", type=int, default=100)
    argparser.add_argument("--seed", type=int, default=42)
    argparser.add_argument("--sentence_output", type=str, default="seisyun.csv")
    argparser.add_argument("--comparison_output", type=str, default="comparison.csv")
    
    args = argparser.parse_args()

    main(args)