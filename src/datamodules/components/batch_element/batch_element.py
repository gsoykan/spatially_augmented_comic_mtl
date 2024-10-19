from typing import Optional, List, Dict
import pandas as pd

from src.datamodules.components.batch_element.batch_element_type import BatchElementType
from src.datamodules.components.batch_element.panel_in_batch_element import PanelInBatchElement
from tqdm import tqdm


# source: https://www.geeksforgeeks.org/what-is-a-clean-pythonic-way-to-have-multiple-constructors-in-python/
class BatchElement:
    def __init__(self,
                 batch_element_type: BatchElementType,
                 context_panels: List[PanelInBatchElement] = [],
                 answer_panels: Optional[List[PanelInBatchElement]] = None,
                 answer_panel: Optional[PanelInBatchElement] = None,
                 correct_answer: Optional[int] = None):
        self.context_panels = context_panels
        self.answer_panels = answer_panels
        self.answer_panel = answer_panel
        self.correct_answer = correct_answer
        self.batch_element_type = batch_element_type

    @classmethod
    def init_from_original_fold_character_coherence_row(cls, row: Dict):
        book_id = str(row['book_id'])
        page_id = int(row['page_id'])
        context_panels = []
        answer_panels = []
        correct_answer = int(row['correct_answer'])
        answer_panel_id = int(row['answer_panel_id'])
        for i in range(3):
            context_panels.append(PanelInBatchElement.create_panel_from_cloze_row(book_id=book_id,
                                                                                  page_id=page_id,
                                                                                  panel_id=int(row[
                                                                                                   f'context_panel_{i}_id']),
                                                                                  raw_texts=[
                                                                                      row[f'context_text_{i}_0'],
                                                                                      row[f'context_text_{i}_1'],
                                                                                      row[f'context_text_{i}_2']
                                                                                  ]))
        for j in range(2):
            answer_panels.append(PanelInBatchElement.create_panel_from_cloze_row(book_id=str(book_id),
                                                                                 page_id=page_id,
                                                                                 panel_id=answer_panel_id,
                                                                                 raw_texts=[
                                                                                     row[f'answer_candidate_{j}_box_0'],
                                                                                     row[f'answer_candidate_{j}_box_1']
                                                                                 ]))

        # correct answer is always zeroth one
        if correct_answer == 1:
            answer_panels.reverse()
        return cls(batch_element_type=BatchElementType.ORIGINAL_FOLD_CHAR_COHERENCE,
                   context_panels=context_panels,
                   answer_panels=answer_panels,
                   correct_answer=correct_answer,
                   answer_panel=None)

    @classmethod
    def init_from_original_fold_visual_cloze_row(cls, row: Dict):
        book_id = str(row['book_id'])
        page_id = int(row['page_id'])
        context_panels = []
        answer_panels = []
        correct_answer = int(row['correct_answer'])
        for i in range(3):
            context_panels.append(PanelInBatchElement.create_panel_from_cloze_row(book_id=book_id,
                                                                                  page_id=page_id,
                                                                                  panel_id=int(row[
                                                                                                   f'context_panel_{i}_id']),
                                                                                  raw_texts=[
                                                                                      row[f'context_text_{i}_0'],
                                                                                      row[f'context_text_{i}_1'],
                                                                                      row[f'context_text_{i}_2']
                                                                                  ]))
            answer_book_id, answer_page_id, answer_panel_id = list(
                map(lambda x: int(x), row[f'answer_candidate_id_{i}'].split('_')))
            answer_panels.append(PanelInBatchElement.create_panel_from_cloze_row(book_id=str(answer_book_id),
                                                                                 page_id=answer_page_id,
                                                                                 panel_id=answer_panel_id,
                                                                                 raw_texts=[]))
        return cls(batch_element_type=BatchElementType.ORIGINAL_FOLD_VISUAL_CLOZE,
                   context_panels=context_panels,
                   answer_panels=answer_panels,
                   correct_answer=correct_answer,
                   answer_panel=None)

    @classmethod
    def init_from_original_fold_text_cloze_row(cls, row: Dict):
        book_id = str(row['book_id'])
        page_id = int(row['page_id'])
        context_panels = []
        answer_panels = []
        correct_answer = int(row['correct_answer'])
        correct_answer_text = row[f'answer_candidate_{correct_answer}_text']
        assert correct_answer_text, 'correct answer should have text'
        for i in range(3):
            context_panels.append(PanelInBatchElement.create_panel_from_cloze_row(book_id=book_id,
                                                                                  page_id=page_id,
                                                                                  panel_id=int(row[
                                                                                                   f'context_panel_{i}_id']),
                                                                                  raw_texts=[
                                                                                      row[f'context_text_{i}_0'],
                                                                                      row[f'context_text_{i}_1'],
                                                                                      row[f'context_text_{i}_2']
                                                                                  ]))
            panel_id = int(row['answer_panel_id']) if i == correct_answer else None
            answer_panels.append(PanelInBatchElement.create_panel_from_cloze_row(book_id=book_id,
                                                                                 page_id=page_id,
                                                                                 panel_id=panel_id,
                                                                                 raw_texts=[
                                                                                     row[
                                                                                         f'answer_candidate_{i}_text']
                                                                                 ]))
        return cls(batch_element_type=BatchElementType.ORIGINAL_FOLD_TEXT_CLOZE,
                   context_panels=context_panels,
                   answer_panels=answer_panels,
                   correct_answer=correct_answer,
                   answer_panel=None)

    @classmethod
    def init_from_custom_json(cls, custom_json: List[Dict],
                              book_id: str,
                              max_context_panel_count: Optional[int] = None):
        if max_context_panel_count is not None:
            custom_json = custom_json[-(max_context_panel_count + 1):]
        elements = list(map(lambda x: PanelInBatchElement.create_panel_from_custom_json(x, book_id), custom_json))
        answer_panel = elements.pop()
        context_panels = elements
        return cls(batch_element_type=BatchElementType.CUSTOM_JSON,
                   context_panels=context_panels,
                   answer_panels=None,
                   answer_panel=answer_panel,
                   correct_answer=None)


# TODO: @gsoykan turn those into tests
####################################################################################################

def text_cloze_check():
    text_cloze_csv_path = '../../../../data/original_folds/text_cloze_dev_easy.csv'
    text_cloze_csv = pd.read_csv(text_cloze_csv_path)
    text_cloze_csv = text_cloze_csv.reset_index()
    batch_elements = []
    for index, row in tqdm(text_cloze_csv.iterrows()):
        b_e = BatchElement.init_from_original_fold_text_cloze_row(row)
        batch_elements.append(b_e)
        # if index > 10:
        #    break
    print(text_cloze_csv.head())


def visual_cloze_check():
    visual_cloze_csv_path = '../../../../data/original_folds/visual_cloze_dev_easy.csv'
    visual_cloze_csv = pd.read_csv(visual_cloze_csv_path)
    visual_cloze_csv = visual_cloze_csv.reset_index()
    batch_elements = []
    for index, row in tqdm(visual_cloze_csv.iterrows()):
        b_e = BatchElement.init_from_original_fold_visual_cloze_row(row)
        batch_elements.append(b_e)
        # if index > 10:
        #   break
    print(visual_cloze_csv.head())


def char_coherence_check():
    char_coherence_csv_path = '../../../../data/original_folds/char_coherence_dev.csv'
    char_coherence_csv = pd.read_csv(char_coherence_csv_path)
    char_coherence_csv = char_coherence_csv.reset_index()
    batch_elements = []
    for index, row in tqdm(char_coherence_csv.iterrows()):
        b_e = BatchElement.init_from_original_fold_character_coherence_row(row)
        batch_elements.append(b_e)
        if index > 10:
            break
    print(char_coherence_csv.head())


if __name__ == '__main__':
    char_coherence_check()
    # visual_cloze_check()
    # text_cloze_check()
