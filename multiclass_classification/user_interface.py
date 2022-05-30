from evaluate import evaluate
from transformers import logging


def main():
    logging.set_verbosity_error()
    model_path = 'D:/BMSTU/Master/sem_3-4/НИР/py/multiclass_classification/sbert epoch = 1-val_loss = 0.6856-LR = 1.0e-06.pt'
    input_sentence = "Лишившись такого крупного заказчика, компания, и раньше из-за повышения налогов испытывавшая серьезные проблемы, может оказаться на грани банкротства, сообщают аналитики сепулька-информ"
    input_entity = "компания"
    print(evaluate(model_path, input_sentence, input_entity))
    input_sentence = "Путин подписал закон, повышающий штрафы за нарушение правил пожарной безопасности в лесахм"
    input_entity = "Путин"
    print(evaluate(model_path, input_sentence, input_entity))
    input_sentence = "Игорь Андреев получил звание «Заслуженный тренер России»"
    input_entity = "Игорь Андреев"
    print(evaluate(model_path, input_sentence, input_entity))
    input_sentence =  "В 2018 году Россия обыграла Испанию в чемпионате мира по футболу, но, к сожалению, заняла только десятое место в общей таблице"
    input_entity = "Россия"
    print(evaluate(model_path, input_sentence, input_entity))
    return 1


if __name__ == '__main__':
    main()
