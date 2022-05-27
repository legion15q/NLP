import pandas as pd
import os
import warnings
from itertools import chain
import numpy as np
from ast import literal_eval


class CreateDataset(object):
    def __init__(self, files_dir='work_dir'):
        if files_dir == 'work_dir':
            self.files_dir = os.getcwd() + '/sentiment_dataset2'
        else:
            self.files_dir = files_dir
        self.df = pd.DataFrame()

    def read_dataframe(self):
        return pd.read_csv(os.getcwd() + '\data' + '.csv', encoding='utf-8')

    def create_dataframe(self):
        self.df = pd.DataFrame(
            columns={'doc_id', 'sentence_id', 'sentence', 'entity_type', 'entity_id', 'entity_start_pos',
                     'entity_end_pos', 'entity', 'tonality',
                     'relation_type', 'relation'})
        all_files = os.listdir(self.files_dir)
        ann_files = sorted(list(filter(lambda x: x[-4:] == '.ann', list(all_files))))
        txt_files = sorted(list(filter(lambda x: x[-4:] == '.txt', list(all_files))))
        entities = {'PERSON', 'COUNTRY', 'ORGANIZATION', 'AUTHOR_NEG', 'AUTHOR_POS',
                    'OPINION_WORD_NEG', 'OPINION_WORD_POS', 'ARGUMENT_NEG', 'ARGUMENT_POS'}
        relations = {'POSITIVE_TO', 'NEGATIVE_TO', 'POS_AUTHOR_FROM',
                     'NEG_AUTHOR_FROM'}  # OPINION_RELATES_TO нужно добавить
        relations_map = {'POSITIVE_TO': 1, 'NEGATIVE_TO': -1, 'POS_AUTHOR_FROM': 1, 'NEG_AUTHOR_FROM': -1,
                         'OPINION_WORD_NEG': -1, 'OPINION_WORD_POS': 1, 'ARGUMENT_NEG': -1, 'ARGUMENT_POS': 1}
        opinion_relates_to_count = 0
        opinion_word_neg_count = 0
        opinion_word_pos_count = 0
        argument_neg_count = 0
        argument_pos_count = 0
        for file_index in range(len(ann_files)):
            # неправильно размеченный документ
            if txt_files[file_index] == '15597_text.txt':
                continue
            data = pd.read_csv(self.files_dir + '/' + ann_files[file_index], sep='\t',
                               names=['RT_type', 'RT_with_pos', 'entity_if_T'],
                               encoding='utf-8')
            for index, row in data.iterrows():
                if row['RT_type'][0] == 'T':
                    T_with_pos = row['RT_with_pos'].split()
                    if T_with_pos[0] in entities:
                        [sentence_id, sentence, start_pos, end_pos] = self.get_sentence_id(T_with_pos[1::],
                                                                                           txt_files[file_index])
                        entities_df = {'doc_id': txt_files[file_index], 'sentence_id': sentence_id,
                                       'sentence': sentence,
                                       'entity_type': T_with_pos[0], 'entity_id': row['RT_type'],
                                       'entity': row['entity_if_T'], 'entity_start_pos': start_pos,
                                       'entity_end_pos': end_pos,
                                       'tonality': [], 'relation_type': [],
                                       'relation': []}
                        self.df.loc[len(self.df.index)] = entities_df

            for index, row in data.iterrows():
                if row['RT_type'][0] == 'R':
                    relation = row['RT_with_pos'].split()
                    relation[1] = relation[1].partition(':')[2]
                    relation[2] = relation[2].partition(':')[2]
                    # print(txt_files[file_index])
                    entity_frame_1 = self.df[
                        (self.df['entity_id'] == relation[1]) & (self.df['doc_id'] == txt_files[file_index])]
                    entity_frame_2 = self.df[
                        (self.df['entity_id'] == relation[2]) & (self.df['doc_id'] == txt_files[file_index])]
                    if not entity_frame_1.empty:
                        if relation[0] not in {'POSITIVE_TO', 'NEGATIVE_TO'}:
                            self.df.loc[entity_frame_1.index[0], 'relation_type'].append(relation[0])
                            self.df.loc[entity_frame_1.index[0], 'relation'].append({relation[1]: relation[2]})
                            if relation[0] in relations:
                                self.df.loc[entity_frame_1.index[0], 'tonality'].append(relations_map[relation[0]])
                    if not entity_frame_2.empty:
                        self.df.loc[entity_frame_2.index[0], 'relation_type'].append(relation[0])
                        self.df.loc[entity_frame_2.index[0], 'relation'].append({relation[1]: relation[2]})
                        if relation[0] in relations:
                            self.df.loc[entity_frame_2.index[0], 'tonality'].append(relations_map[relation[0]])
                        if not entity_frame_1.empty:
                            if relation[0] == 'OPINION_RELATES_TO':
                                rel = entity_frame_1['entity_type'].values[0]
                                try:
                                    self.df.loc[entity_frame_2.index[0], 'tonality'].append(relations_map[rel])
                                    opinion_relates_to_count += 1
                                    if rel == 'OPINION_WORD_NEG':
                                        opinion_word_neg_count += 1
                                    elif rel == 'OPINION_WORD_POS':
                                        opinion_word_pos_count += 1
                                    elif rel == 'ARGUMENT_NEG':
                                        argument_neg_count += 1
                                    elif rel == 'ARGUMENT_POS':
                                        argument_pos_count += 1
                                except:
                                    continue

        count_pos_entities = 0
        count_neg_entities = 0
        count_neutral_entities = 0
        count_neg_pos_entities = 0
        self.combine_author_pos_with_entity()
        for index, _ in self.df.iterrows():
            if not self.df['tonality'].iloc[index]:
                self.df['tonality'].iloc[index] = [0]
            if self.df['tonality'].iloc[index] != [0] and sum(self.df['tonality'].iloc[index]) == 0:
                self.df['tonality'].iloc[index] = [-2]
            if not self.df['relation_type'].iloc[index]:
                self.df['relation_type'].iloc[index] = [None]
            if not self.df['relation'].iloc[index]:
                self.df['relation'].iloc[index] = [None]
            if sum(self.df['tonality'].iloc[index]) < 0 and self.df['tonality'].iloc[index] != [-2]:
                self.df['tonality'].iloc[index] = [-1]
            if sum(self.df['tonality'].iloc[index]) > 0:
                self.df['tonality'].iloc[index] = [1]
            self.df['tonality'].iloc[index] = self.df['tonality'].iloc[index][0]
            if self.df['entity_type'].iloc[index] in {'PERSON', 'COUNTRY', 'ORGANIZATION'}:
                if self.df['tonality'].iloc[index] == 1:
                    count_pos_entities += 1
                if self.df['tonality'].iloc[index] == -1:
                    count_neg_entities += 1
                if self.df['tonality'].iloc[index] == 0:
                    count_neutral_entities += 1
                if self.df['tonality'].iloc[index] == -2:
                    count_neg_pos_entities += 1
        print('count_pos_entities: ', count_pos_entities, '\n', 'count_neg_entities: ', count_neg_entities, '\n',
              'count_neutral_entities: ', count_neutral_entities, '\n', 'count_neg_pos_entities: ',
              count_neg_pos_entities, '\n', 'opinion_relates_to_count: ', opinion_relates_to_count,
              '\n', 'opinion_word_neg_count: ', opinion_word_neg_count, '\n', 'opinion_word_pos_count: ',
              opinion_word_pos_count,
              '\n', 'argument_neg_count: ', argument_neg_count, '\n', 'argument_pos_count:', argument_pos_count)
        self.df.to_csv(os.getcwd() + '/' + 'data' + '.csv', encoding='utf-8', index=False)

    def combine_author_pos_with_entity(self):
        relevant_entity = {'PERSON', 'COUNTRY', 'ORGANIZATION'}
        not_relevant_entity = {'AUTHOR_NEG', 'AUTHOR_POS'}

        for index, row in self.df.iterrows():
            doc_id = row['doc_id']
            entity_type = row['entity_type']
            entity_start_pos = row['entity_start_pos']
            entity_end_pos = row['entity_end_pos']
            if entity_type in relevant_entity:
                not_relevant_entity_frame = self.df[
                    (self.df['entity_start_pos'] == entity_start_pos) & (self.df['doc_id'] == doc_id) & (
                            self.df['entity_end_pos'] == entity_end_pos) & (
                            (self.df['entity_type'] == 'AUTHOR_NEG') | (self.df['entity_type'] == 'AUTHOR_POS'))]
                if not not_relevant_entity_frame.empty:
                    self.df.loc[index, 'relation_type'].extend(
                        chain(list(chain(*not_relevant_entity_frame['relation_type']))))
                    self.df.loc[index, 'relation'].extend(list(chain(*not_relevant_entity_frame['relation'])))
                    self.df.loc[index, 'tonality'].extend(list(chain(*not_relevant_entity_frame['tonality'])))

    def get_sentence_id(self, positions, file_name):
        print(positions)
        start_pos = int(positions[0])
        end_pos = int(positions[1].split(';')[0])
        file = open(self.files_dir + '/' + file_name, 'r', encoding='utf8')
        document = file.read()
        file.close()
        doc_sentences = document.split('.')
        sentence_id = None
        sentence = None
        for i in range(len(doc_sentences)):
            sentence_len = len(doc_sentences[i])
            sentence = doc_sentences[i]
            sentence_id = i
            if end_pos < sentence_len:
                return [sentence_id, sentence, start_pos, end_pos]
            else:
                end_pos = end_pos - len(doc_sentences[i]) - 1
                start_pos = start_pos - len(doc_sentences[i]) - 1
        return [sentence_id, sentence, start_pos, end_pos]


def main():
    warnings.filterwarnings("ignore")
    RF = CreateDataset()
    RF.create_dataframe()
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_colwidth', 199)
    pd.set_option('display.width', None)
    # pd.set_option('display.max_rows', None)
    # df = RF.read_dataframe()
    # df_without_sentence = df.loc[:, df.columns != 'sentence']
    # print(df_without_sentence)

    return 1


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
