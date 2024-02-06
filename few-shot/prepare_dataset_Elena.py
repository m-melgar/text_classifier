import json
import os.path
from glob import glob

__LABELS = ("Negative", "Neutral", "Positive")


def get_segments(jsonfile_: str) -> list:
    with open(jsonfile_, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if 0.4 < data["response"]["sentiment"] < 0.7:  # only accept high or low sentiment transcripts
        return []
    segments_ = [i["transcript"] for i in data["response"]["segments"]]
    # sentiment_old = [__LABELS[i["sentiment"]] for i in data["response"]["segments"]]
    return segments_


if __name__ == '__main__':

    jsonfiles = glob(r"C:\Users\melsegma\Downloads\Downloads\*.json")
    results_path = r"C:\Users\melsegma\PycharmProjects\text_classifier\few-shot\txtlabels"
    for jsonfile in jsonfiles:
        segments = get_segments(jsonfile)
        if segments:
            _, txt_name = os.path.split(jsonfile)
            segments = ["{}\n".format(segment) for segment in segments]
            with open(os.path.join(results_path, txt_name + ".txt"), 'w', encoding='utf-8') as fp:
                fp.writelines(segments)

    print("DONE!")
