from evaluate import evaluate
from transformers import logging


def main():
    logging.set_verbosity_error()
    model_path = 'D:/BMSTU/Master/sem_3-4/НИР/py/multilabel_classification/sbert epoch = 0-val_loss = 0.5346-LR = 1.0e-06.pt'
    input_sentence = "Лишившись такого крупного заказчика, компания, и раньше испытывавшая серьезные проблемы из-за повышения налогов, может оказаться на грани банкротства, сообщают аналитики сепулька-информ"
    input_entity = "компания"
    print(evaluate(model_path, input_sentence, input_entity))
    input_sentence = "Лишившись такого крупного заказчика, компания, и раньше испытывавшая серьезные проблемы из-за повышения налогов, может оказаться на грани банкротства, сообщают аналитики сепулька-информ"
    input_entity = "заказчика"
    print(evaluate(model_path, input_sentence, input_entity))
    input_sentence = "Игорь Андреев получил звание «Заслуженный тренер России»"
    input_entity = "Игорь Андреев"
    print(evaluate(model_path, input_sentence, input_entity))
    input_sentence = "К сожалению, Краснодар только однажды обыграл Зенит, но получил за это признание"
    input_entity = "Краснодар"
    print(evaluate(model_path, input_sentence, input_entity))
    return 1


if __name__ == '__main__':
    main()
