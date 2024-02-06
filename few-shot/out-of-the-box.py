from transformers import pipeline

if __name__ == '__main__':
    classifier = pipeline(model="finiteautomata/bertweet-base-sentiment-analysis")
    # pred = classifier(
    #     ["me ha dicho tu compañera que ha habido un problema pero que ya estaba solucionado",
    #      "me ha dicho tu compañera que ha habido un problema pero que ya estaba solucionado pero sigue sin dejarme",
    #      "me ha dicho tu compañera no tiene noticias sobre el problema",
    #      "me ha dicho tu compañera no tiene noticias sobre el problema pero que ayer se resolvió "])
    pred = classifier(
        ["Porque estamos pendientes, el otro día nos mandó esto bastante material y aún faltaba entregar una caja "])
    # print(classifier.model)
    print(pred)
