import pandas as pd
import method_with_dict

def evaluate(thesaurus, input_sentence, input_entity):
    data = pd.DataFrame(columns=['sentence', 'entity_start_pos', 'entity_end_pos', 'entity', 'entity_type', 'tonality'])
    e_s_p = input_sentence.find(input_entity)
    e_e_p = e_s_p + len(input_entity)
    data.loc[len(data.index)] = {'sentence' : input_sentence, 'entity_start_pos' : e_s_p, 'entity_end_pos' : e_e_p,
                                 'entity' : input_entity, 'entity_type' : 'None', 'tonality' : 'None'}
    eval_words = method_with_dict.evaluate(data)
    method_with_dict.Solve_ambiguity(thesaurus, eval_words, 'w', True)
    return 1

def main():
    thesaurus = method_with_dict.read_thesaurus()
    input_sentence = "Лишившись такого крупного заказчика, компания, и раньше испытывавшая серьезные проблемы из-за повышения налогов, может оказаться на грани банкротства, сообщают аналитики сепулька-информ"
    input_entity = "компания"
    evaluate(thesaurus, input_sentence, input_entity)
    input_sentence = "Лишившись такого крупного заказчика, компания, и раньше испытывавшая серьезные проблемы из-за повышения налогов, может оказаться на грани банкротства, сообщают аналитики сепулька-информ"
    input_entity = "заказчика"
    evaluate(thesaurus, input_sentence, input_entity)
    input_sentence = "Игорь Андреев получил звание «Заслуженный тренер России»"
    input_entity = "Игорь Андреев"
    evaluate(thesaurus, input_sentence, input_entity)
    input_sentence = "К сожалению, Краснодар только однажды обыграл Зенит, но получил за это признание"
    input_entity = "Краснодар"
    evaluate(thesaurus, input_sentence, input_entity)
    return 1


if __name__ == '__main__':
    main()