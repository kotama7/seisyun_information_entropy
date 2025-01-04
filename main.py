import pandas as pd
import argparse
import os

from module.seisyun import seisyun
from module.set_seed import set_seed
from module.paired_comparison import paired_comparison
from module.plot import plot
from module.calculate_order import calculate_goodness

def main(args):

    set_seed(args.seed)

    if not os.path.exists(args.sentence_output):

        seisyun_dict = {}

        while len(seisyun_dict) < args.sentence_num:
            score, sentence = seisyun(args.mask_num)
            seisyun_dict[sentence] = score

        df = pd.DataFrame(seisyun_dict.items(), columns=["sentence", "score"]).sort_values("score", ascending=False)

        df.to_csv(args.sentence_output, index=False)

    else:
    
        df = pd.read_csv(args.sentence_output, index_col=False)

    # Use only the top N sentences
    df = df.head(args.use_sentence_num+args.remove_top_n)

    if not os.path.exists(args.comparison_output):

        comparison_df = paired_comparison(df, args.seed, args.evaluater_num)

        comparison_df.to_csv(args.comparison_output, index=False)

    else:
        
        comparison_df = pd.read_csv(args.comparison_output, index_col=False)

    if not os.path.exists(args.goodness_output):

        goodness_df = calculate_goodness(comparison_df, args.use_sentence_num, args.evaluater_num, args.remove_top_n)

        goodness_df.to_csv(args.goodness_output, index=False)

    else:

        goodness_df = pd.read_csv(args.goodness_output, index_col=False)

    df = df.loc[args.remove_top_n:,:]

    plot(df["score"], goodness_df, args.plot_output)


if __name__ == "__main__":
    
    argparser = argparse.ArgumentParser()

    argparser.add_argument("--sentence_num", type=int, default=1000)
    argparser.add_argument("--mask_num", type=int, default=10)
    argparser.add_argument("--use_sentence_num", type=int, default=30)
    argparser.add_argument("--remove_top_n", type=int, default=4)
    argparser.add_argument("--seed", type=int, default=42)
    argparser.add_argument("--evaluater_num", type=int, default=2)
    argparser.add_argument("--sentence_output", type=str, default="seisyun.csv")
    argparser.add_argument("--comparison_output", type=str, default="comparison.csv")
    argparser.add_argument("--goodness_output", type=str, default="goodness.csv")
    argparser.add_argument("--plot_output", type=str, default="plot.png")
    

    args = argparser.parse_args()

    main(args)