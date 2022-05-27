import pandas as pd
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import re
import pymorphy2
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

model = SentenceTransformer('sberbank-ai/sbert_large_nlu_ru')


def Calc_SBERT_Similarity(lhs, rhs):
    sentence_embeddings = model.encode([lhs, rhs])
    scores = np.zeros((sentence_embeddings.shape[0], sentence_embeddings.shape[0]))
    for i in range(sentence_embeddings.shape[0]):
        scores[i, :] = cosine_similarity(
            [sentence_embeddings[i]],
            sentence_embeddings
        )[0]
    return scores[0][1]


def write_to_csv(DF, DF_name):
    DF.to_csv(os.getcwd() + '/' + DF_name + '.csv', encoding='utf-8', index=False)
    return DF


# h=50 -- оптимальное значение
def read_evaluation_words(rw='w', dir='/data.csv', h=70) -> pd.DataFrame:
    new_df = pd.DataFrame()
    if rw == 'w':
        data = pd.read_csv(os.getcwd() + dir)
        options = ['PERSON', 'ORGANIZATION', 'COUNTRY']
        data = data[data['entity_type'].isin(options)]
        count = 0
        to_delete = []
        for index, row in data.iterrows():
            entity_start_pos = int(row['entity_start_pos'])
            if entity_start_pos < 0:
                count += 1
                to_delete.append(index)
        print(len(data))
        data = data.drop(labels=to_delete, axis=0)
        print(len(data))
        y = list(data["tonality"])
        indexes = list(data.index)
        indexes_train, indexes_test, y_train, y_test = train_test_split(indexes, y,
                                                                        train_size=0.65,
                                                                        random_state=42,
                                                                        stratify=y)
        count = 7
        indexes_new_train = []
        y_new_train = []
        for i in range(len(y_train)):
            if count != 0 and y_train[i] == -2:
                y_test.append(-2)
                count -= 1
                indexes_test.append(indexes_train[i])
            else:
                indexes_new_train.append(indexes_train[i])
                y_new_train.append(y_train[i])
        indexes_train = indexes_new_train
        y_train = y_new_train
        # перемешиваем
        indexes_train, y_train = shuffle(indexes_train, y_train, random_state=42)
        indexes_test, y_test = shuffle(indexes_test, y_test, random_state=42)
        indexes_test, indexes_val, y_test, y_val = train_test_split(indexes_test, y_test,
                                                                    train_size=0.3,
                                                                    random_state=42,
                                                                    stratify=y_test)
        data = data.loc[indexes_test]

        new_df = pd.DataFrame(columns=['sentence', 'evaluation_words', 'tonality', 'entity', 'entity_type'])
        morph = pymorphy2.MorphAnalyzer()
        for index, row in data.iterrows():
            sentence = row['sentence']
            entity_start_pos = int(row['entity_start_pos'])
            entity_end_pos = int(row['entity_end_pos'])
            if h != None:
                s_h = entity_start_pos - h
                if s_h < 0:
                    s_h = 0
                e_h = entity_end_pos + h
                if e_h > len(sentence):
                    e_h = len(sentence)
            else:
                s_h = 0
                e_h = len(sentence)
            evaluation_words = sentence[s_h: entity_start_pos] + sentence[entity_end_pos:e_h]
            evaluation_words = re.sub("[^A-Za-zА-Яа-я{} ]", " ", evaluation_words)
            evaluation_words = evaluation_words.lower()
            while '  ' in evaluation_words:
                evaluation_words = evaluation_words.replace('  ', ' ')
            lemmas_evaluation_words = ''
            for i in evaluation_words.split(' '):
                lemmas_evaluation_words += morph.parse(i)[0].normal_form + ' '

            new_df.loc[len(new_df.index)] = {'sentence': sentence, 'evaluation_words': lemmas_evaluation_words,
                                             'tonality': row['tonality'],
                                             'entity': row['entity'], 'entity_type': row['entity_type']}
        write_to_csv(new_df, 'df_for_dict_method')
    elif rw == 'r':
        new_df = pd.read_csv(os.getcwd() + '/df_for_dict_method.csv')
    return new_df


def Solve_ambiguity(thesaurus_df, eval_words_df, rw_mode='w'):
    pred_tonality = []
    negative_particles = [' не ', ' ни ']
    if rw_mode == 'w':
        tonality_map = {'positive': 1, 'negative': -1, 'neutral': 0, 'positive/negative': 0}
        pred_tonality = []
        for index, row in eval_words_df.iterrows():
            sentence = str(row['sentence'])
            tonality = []
            is_negative_particles = False
            for word in str(row['evaluation_words']).split(' '):
                word = ' ' + word.strip() + ' '
                if word in negative_particles:
                    is_negative_particles = True
                lemmas_df = thesaurus_df.loc[(thesaurus_df['Lemma'] == word)]
                if lemmas_df.empty:
                    tonality.append(0)
                else:
                    contexts = lemmas_df['Ambiguity'].to_list()
                    score_list = []
                    for context in contexts:
                        score = 1
                        if str(context) != 'nan':
                            score = Calc_SBERT_Similarity(context, sentence)
                        score_list.append(score)
                    index_of_max_elem = score_list.index(max(score_list))
                    main_context = contexts[index_of_max_elem]
                    if str(main_context) == 'nan':
                        main_tonality = thesaurus_df.loc[(thesaurus_df['Lemma'] == word)]['Sentiment'].values[0]
                    else:
                        main_tonality = \
                            thesaurus_df.loc[
                                (thesaurus_df['Ambiguity'] == main_context) & (thesaurus_df['Lemma'] == word)][
                                'Sentiment'].values[0]
                    if is_negative_particles:
                        tonality.append(tonality_map[main_tonality.strip()] * (-1))
                    else:
                        tonality.append(tonality_map[main_tonality.strip()])
            sum_ = sum(tonality)
            ton = 0
            if sum_ < 0:
                ton = -1
            elif sum_ > 0:
                ton = 1
            pred_tonality.append(ton)
        write_to_csv(pd.DataFrame(pred_tonality), 'predict_df')
    elif rw_mode == 'r':
        pred_tonality = pd.read_csv('predict_df.csv').values
    true_tonality = list(eval_words_df['tonality'])
    print(metrics.classification_report(true_tonality, pred_tonality,
                                        target_names=["neg-pos", "negative", "neutral", "positive"]))
    accuracy_per_class(pred_tonality, true_tonality)
    return 1


def accuracy_per_class(preds, labels):
    preds_flat = np.array(preds).flatten()
    labels_flat = np.array(labels).flatten()
    k = 0
    labels = ["neg-pos", "negative", "neutral", "positive"]
    for label in np.unique(labels_flat):
        y_preds = preds_flat[labels_flat == label]
        y_true = labels_flat[labels_flat == label]
        print(f'\nClass: {labels[k]}')
        print(f'\nAccuracy: {len(y_preds[y_preds == label])}/{len(y_true)}\n')
        k += 1


def read_thesaurus(dir='/rusentilex_2019_clear.txt'):
    print(os.getcwd() + dir)
    df = pd.read_csv(os.getcwd() + dir, sep=",", names=['word', 'TAG', 'Lemma', 'Sentiment', 'Source', 'Ambiguity'],
                     encoding='utf8', on_bad_lines='skip')
    df['Ambiguity'].replace(regex=True, inplace=True, to_replace=r'\"', value=r'')
    lemmas = df['Lemma'].to_list()
    edit_lemmas = []
    for i in range(len(lemmas)):
        edit_lemmas.append(' ' + lemmas[i].strip(' ') + ' ')
    df['Lemma'] = edit_lemmas
    return df


def main():
    thesaurus = read_thesaurus()
    eval_words_df = read_evaluation_words('w')
    print(thesaurus)
    print(eval_words_df)
    Solve_ambiguity(thesaurus, eval_words_df, 'w')
    '''
    1) Как-то выделить оценочные слова. Самый простой вариант это считать что все слова, кроме целевого, являются
    оценочными
    2) Пройтись для каждого оценочного слова по словарю rusentilex и посмотреть есть ли тональность для этого слова
    2.1) Если для этого слова несколько вариантов контекста, то с помощью BERT определить какой контекст подходит
    наиболее всего (соответственно и тональность в этом контексте)
    3) Если в словаре нет данных слов и/или они все имеют нейтральную тональность, то тональность целевого слова = 0
    4) Если есть хотя бы одна положительная тональность, то нет отрицательной, то тональность целевого слова = 1, и
    наоборот
    5) Если есть и положительная и отрицательная тональность, то тональность целевого слова = -2
    '''

    return 1


if __name__ == '__main__':
    main()
