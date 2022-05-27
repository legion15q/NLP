import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
import re
from sklearn.utils import shuffle


def main():
    data = pd.read_csv('data.csv')
    new_df = pd.DataFrame(columns=['sentence', 'tonality', 'entity', 'entity_type'])
    count = 0
    for index, row in data.iterrows():
        sentence = row['sentence']
        entity_start_pos = int(row['entity_start_pos'])
        entity_end_pos = int(row['entity_end_pos'])
        if entity_start_pos < 0:
            count += 1
            continue
        sentence_with_x = sentence[: entity_start_pos] + "{X}" + sentence[entity_end_pos:]
        sentence_with_x = re.sub("[^A-Za-zА-Яа-я{} ]", "", sentence_with_x)
        start_position = sentence_with_x.find("{X}")
        end_position = start_position + 3
        sentence_with_x = sentence_with_x[: start_position] + " {X} " + sentence_with_x[end_position:]
        while '  ' in sentence_with_x:
            sentence_with_x = sentence_with_x.replace('  ', ' ')

        sentence_with_x = sentence_with_x.lower()
        new_df.loc[len(new_df.index)] = {'sentence': sentence_with_x, 'tonality': row['tonality'],
                                         'entity': row['entity'], 'entity_type': row['entity_type']}

    print("count of skip incorrect sentence", count)
    options = ['PERSON', 'ORGANIZATION', 'COUNTRY']
    new_df = new_df[new_df['entity_type'].isin(options)]
    print('Общее число предложений:', len(new_df))
    # new_df = new_df.drop(new_df[new_df.tonality == -2].index)
    # print(len(new_df.loc[new_df['tonality'] == -2]))
    # сначала перемешаем датасет
    new_df = new_df.sample(frac=1, random_state=42)
    # затем разделим его на train val и test
    X = list(new_df["sentence"])
    y = list(new_df["tonality"])
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        train_size=0.65,
                                                        random_state=42,
                                                        stratify=y)
    # перекидываем 10 значений -2 класса в test_data
    # print(len(X_train), len(y_train))
    # print(len(X_test), len(y_test))
    count = 7
    X_new_train = []
    y_new_train = []
    for i in range(len(y_train)):
        if count != 0 and y_train[i] == -2:
            y_test.append(-2)
            count -= 1
            X_test.append(X_train[i])
        else:
            X_new_train.append(X_train[i])
            y_new_train.append(y_train[i])
    X_train = X_new_train
    y_train = y_new_train
    # перемешиваем
    X_train, y_train = shuffle(X_train, y_train, random_state=42)
    X_test, y_test = shuffle(X_test, y_test, random_state=42)

    # print(len(X_train), len(y_train))
    # print(len(X_test), len(y_test))

    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test,
                                                    train_size=0.3,
                                                    random_state=42,
                                                    stratify=y_test)

    # прибавляем 2 т.к. элементы (классы) не могут быть отрицательным в Numpy
    print(f"Количество строк в y_train по классам: {np.bincount([item + 2 for item in y_train])}")
    print(f"Количество строк в y_test по классам: {np.bincount([item + 2 for item in y_test])}")
    print(f"Количество строк в y_val по классам: {np.bincount([item + 2 for item in y_val])}")

    # создадим датафремы из списков
    train_df = pd.DataFrame({'sentence': X_train, 'tonality': y_train})
    test_df = pd.DataFrame({'sentence': X_test, 'tonality': y_test})
    val_df = pd.DataFrame({'sentence': X_val, 'tonality': y_val})

    # upsampling train и val датасета
    main_datasets = [train_df, val_df]
    low_class_labels = [1, -1, -2]
    # low_class_labels = [1, -1]
    for i in range(len(main_datasets)):
        for label in low_class_labels:
            df = main_datasets[i]
            rat = len(df.loc[df['tonality'] == 0]) // len(df.loc[df['tonality'] == label])
            # возможно надо чуть-чуть занизить все классы кроме нулевого
            df_temp = df.loc[df['tonality'] == label]
            df_temp = df_temp.loc[df_temp.index.repeat(rat)]
            df = pd.concat([df, df_temp]).sample(frac=1, random_state=42)
            main_datasets[i] = df
    train_df, val_df = main_datasets
    len_train_0 = len(train_df.loc[train_df['tonality'] == 0])
    len_val_0 = len(val_df.loc[val_df['tonality'] == 0])

    # preprocessing for multilabelling
    all_datasets = [train_df, val_df, test_df]
    for i in range(len(all_datasets)):
        df = all_datasets[i]
        df['neutral'] = np.where(df.tonality == 0, 1, 0)
        df['positive'] = np.where(df.tonality == 1, 1, 0)
        df['negative'] = np.where(df.tonality == -1, 1, 0)
        df.loc[df.tonality == -2, ('negative', 'positive')] = 1
        df.pop('tonality')
        all_datasets[i] = df
    train_df, val_df, test_df = all_datasets

    # save to files
    train_df.to_csv(os.getcwd() + '/train_dataset.csv', encoding='utf-8', index=False)
    test_df.to_csv(os.getcwd() + '/test_dataset.csv', encoding='utf-8', index=False)
    val_df.to_csv(os.getcwd() + '/valid_dataset.csv', encoding='utf-8', index=False)

    return 1


if __name__ == '__main__':
    main()
