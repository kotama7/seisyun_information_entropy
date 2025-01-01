import random
from transformers import AutoTokenizer,  AutoModelForSequenceClassification
import torch
import numpy as np
from scipy.stats import gmean
from transformers import pipeline


def seisyun():
    recur = random.randint(1, 10)
    K_RANGE = 100

    masks = ""

    for i in range(recur):
        masks += "[MASK]"

    unmasker = pipeline('fill-mask', model='tohoku-nlp/bert-base-japanese', device="mps")

    sentence = f"私は{masks}をします。"
    prob_ls = []

    for i in range(recur):
        candidate_ls = unmasker(sentence, top_k=K_RANGE)

        # The first sentence is selected if candidate sentences are multiple
        if recur - i != 1:
            select_ind = 0
            candidate_ls = candidate_ls[select_ind]

        score_ls = [candidate['score'] for candidate in candidate_ls]
        sum_score = sum(score_ls)
        score_ls = [score / sum_score for score in score_ls]
        select_ind = random.choices([i for i in range(len(score_ls))], weights=score_ls)[0]

        print(select_ind)

        sentence = candidate_ls[select_ind]['sequence']
        sentence = sentence.replace("[CLS]", "").replace("[SEP]", "")
        prob_ls.append(candidate_ls[select_ind]['score'])
        print(sentence)

    prob = gmean(prob_ls)
    print("Sentence entropy:", prob)

    text_classifier = pipeline("text-classification", model='koheiduck/bert-japanese-finetuned-sentiment', device="mps")
    negaposi = text_classifier(sentence, top_k=3)

    for i in range(3):
        if negaposi[i]['label'] == 'POSITIVE':
            negaposi_score = negaposi[i]['score']
            print("negaposi score:", negaposi_score)
            break

    
    tokenizer = AutoTokenizer.from_pretrained("line-corporation/line-distilbert-base-japanese", trust_remote_code=True)
    model =  AutoModelForSequenceClassification.from_pretrained("liwii/fluency-score-classification-ja").to("mps")

    input_tokens = tokenizer(
        sentence,
        return_tensors='pt',
        padding=True).to("mps")

    output = model(**input_tokens)

    with torch.no_grad():
        # Probabilities of [not_fluent, fluent]
        probs = torch.nn.functional.softmax(
            output.logits, dim=1)
        sentence_probability = probs[:,1].item()

    print("Sentence fluent:", sentence_probability)

    score = -np.log(prob) - np.log((1-negaposi_score)) - np.log((1-sentence_probability))

    print("Score:", score)

    return score, sentence