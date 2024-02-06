import itertools
import json
import os.path
from glob import glob

import pandas as pd
from tqdm import tqdm
from transformers import pipeline

# good models

""" STARTS
LiYuan/amazon-review-sentiment-analysis
nlptown/bert-base-multilingual-uncased-sentiment
mrm8488/electricidad-base-finetuned-muchocine
EstherT/clasificador-muchocine
marianna13/bert-multilingual-sentiment
"""
__LABELS = ("Negative", "Neutral", "Positive")
model_list = ["Manauu17/roberta_sentiments_es",
              "Manauu17/enhanced_roberta_sentiments_es",
              "edumunozsala/RuPERTa_base_sentiment_analysis_es",
              "finiteautomata/beto-sentiment-analysis",  # esta gusta
              "citizenlab/twitter-xlm-roberta-base-sentiment-finetunned",  # esta gusta tb
              "rohanrajpal/bert-base-en-hi-codemix-cased",
              ]
model_list_stars = ['LiYuan/amazon-review-sentiment-analysis',
                    'nlptown/bert-base-multilingual-uncased-sentiment',
                    'mrm8488/electricidad-base-finetuned-muchocine',  # gusta
                    'EstherT/clasificador-muchocine',  # gusta
                    'marianna13/bert-multilingual-sentiment']

model_list_english = ["cardiffnlp/twitter-roberta-base-sentiment-latest",
                      "finiteautomata/bertweet-base-sentiment-analysis",  # buena
                      "sbcBI/sentiment_analysis_model",
                      "j-hartmann/sentiment-roberta-large-english-3-classes",
                      "sbcBI/sentiment_analysis",
                      "ahmedrachid/FinancialBERT-Sentiment-Analysis",
                      "citizenlab/twitter-xlm-roberta-base-sentiment-finetunned",
                      "Souvikcmsa/BERT_sentiment_analysis",
                      "FinanceInc/auditor_sentiment_finetuned",

                      ]


def get_segments(jsonfile: str) -> tuple[list, list]:
    with open(jsonfile, 'r', encoding='utf-8') as f:
        data = json.load(f)
    segments = [i["transcript"] for i in data["response"]["segments"]]
    sentiment_old = [i["sentiment"] for i in data["response"]["segments"]]
    return segments, sentiment_old


def group_by_speakerTag(segments2group: list[dict]) -> list[str]:
    """
    Agrupa todos los segmentos por speakerTag.
    segment 1: ["dime", speaker 1]                               segment 1: ["dime tu teléfono", speaker 1]
    segment 1: ["tu teléfono", speaker 1]            -------->   segment 2: ["es el seis seis seis seis seis", speaker 2]
    segment 2: ["es el", speaker 2]
    segment 2: ["seis seis seis seis seis", speaker 2]

    ¡¡ ATENCIÓN !!: los nuevos segmentos sólo contienen los campos "transcript" y "words".

    Args:
    :param segments2group: list[dict]
            Segmentos de habla, obtenidos del campo ["segments"] del json.
    """

    new_transcripts = list()

    # Lista con el speaker tag de cada segmento como array, [1 1 1 1 1 2 1 1 2 2 ...]
    speaker_tag_list = [segment["words"][0]["speakerTag"] for segment in segments2group]

    slice_start = 0
    for speakerTag, group in itertools.groupby(speaker_tag_list):
        len_group = len(list(group))

        # concatena el campo transcript de todos los segmentos que coincidan en speakerTag
        concatenated_transcript = "".join(
            [ii["transcript"] for ii in segments2group[slice_start:slice_start + len_group]])

        new_transcripts.append(concatenated_transcript)
        slice_start += len_group  # avanzamos hasta la siguiente posición

    # genera la salida esperada como una lista de diccionarios (como en el .json)

    return [new_transcript for new_transcript in new_transcripts]


def get_segments_grouped_by_speaker(jsonfile: str):
    with open(jsonfile, 'r', encoding='utf-8') as f:
        data = json.load(f)
    segments_data = [i for i in data["response"]["segments"]]

    return group_by_speakerTag(segments_data)


if __name__ == '__main__':

    jsonfiles = glob(r"C:\Users\melsegma\Downloads\TMP\testingles\*.json")
    for model in tqdm(model_list):
        for jsonfile in tqdm(jsonfiles):
            _, json_name = os.path.split(jsonfile)

            results_filename = f"result_sentiments_{json_name}.csv"
            results_path = f"./results/INGLES_{model.replace('/', '_')}"
            os.makedirs(results_path, exist_ok=True)

            results_total_path = os.path.join(results_path, results_filename)

            segments, sentiment_old = get_segments(jsonfile)
            classifier = pipeline(model=model)
            pred = classifier(segments)

            labels = [i["label"] for i in pred]
            scores = [i["score"] for i in pred]

            scores = [str(round(score, 3)).replace(".", ",") for score in scores]

            data = {'segments': segments,
                    'score': scores,
                    'label': labels,
                    'sentiment_old': sentiment_old
                    }
            df = pd.DataFrame(data)
            df.to_csv(results_total_path)
    print("DONE!")
