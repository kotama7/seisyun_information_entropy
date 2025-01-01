import torch

import pandas as pd

from transformers import AutoModelForCausalLM, AutoTokenizer

def paired_comparison(df):

    DEFAULT_SYSTEM_PROMPT = "あなたは誠実で優秀な日本人のアシスタントです。特に指示が無い場合は、常に日本語で回答してください。"

    model_name = "elyza/Llama-3-ELYZA-JP-8B"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
    )
    model.eval()

    n = len(df)

    comparison_df = pd.DataFrame(columns=["sentence1", "sentence2", "result"])

    for i in range(n):
        for j in range(n):

            text = f"""
            以下の二つの文章の内、どちらの方が青春を感じる文章か答えてください。回答は必ずA、Bのどちらかで回答してください。\n
            A:{df.iloc[i,:]["sentence"]}\n
            B:{df.iloc[j,:]["sentence"]}\n
            """

            messages = [
                {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ]

            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            token_ids = tokenizer.encode(
                prompt, add_special_tokens=False, return_tensors="pt"
            )

            with torch.no_grad():
                output_ids = model.generate(
                    token_ids.to(model.device),
                    max_new_tokens=1200,
                    do_sample=True,
                    temperature=0.6,
                    top_p=0.9,
                )

            output = tokenizer.decode(
                output_ids.tolist()[0][token_ids.size(1):], skip_special_tokens=True
            )

            print(output)

            print(f"{df.iloc[i,:]['sentence']} vs {df.iloc[j,:]['sentence']}")
            print(comparison_df.columns)

            temp = pd.Series([df.iloc[i,:]["sentence"], df.iloc[j,:]["sentence"], output])

            comparison_df = pd.concat([comparison_df, temp], ignore_index=True, axis=1)

    return comparison_df