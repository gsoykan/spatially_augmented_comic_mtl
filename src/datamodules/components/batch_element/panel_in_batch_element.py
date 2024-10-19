import math
import os
from typing import Optional, List, Dict, Any

from src.datamodules.components.batch_element.text_in_panel import TextInPanel


class PanelInBatchElement:
    def __init__(self,
                 book_id: Optional[str],
                 page_id: Optional[int],
                 panel_id: Optional[int],
                 texts: Optional[List[TextInPanel]]
                 ):
        self.book_id = book_id
        self.page_id = page_id
        self.panel_id = panel_id
        self.texts = texts

    def __repr__(self):
        return f'{str(self.book_id)} - {str(self.page_id)} - {self.panel_id} - {self.texts}'

    def __eq__(self, other):
        if isinstance(other, PanelInBatchElement):
            return self.book_id == other.book_id \
                   and self.page_id == other.page_id \
                   and self.panel_id == other.panel_id
        return False

    @classmethod
    def create_panel_from_cloze_row(cls,
                                    book_id: str,
                                    page_id: int,
                                    panel_id: Optional[int],
                                    raw_texts: List[Optional[str]]):
        raw_texts = list(filter(lambda x: x and type(x) == str, raw_texts))
        return cls(book_id=book_id,
                   page_id=page_id,
                   panel_id=panel_id,
                   texts=list(map(lambda x: TextInPanel(text=x), raw_texts)))

    @classmethod
    def create_panel_from_custom_json(cls,
                                      custom_json: Dict,
                                      book_id: str):
        texts = custom_json['dialogue'] + custom_json['narration']
        texts = sorted(texts, key=lambda x: x['textbox_no'])
        texts = list(map(lambda x: TextInPanel(text=x['text'],
                                               x1=int(x['x1']),
                                               y1=int(x['y1']),
                                               x2=int(x['x2']),
                                               y2=int(x['y2'])
                                               ), texts))
        page_id, panel_id = custom_json['page'], custom_json['panel']
        return cls(book_id, page_id, panel_id, texts)

    def generate_img_path(self, panel_dir: str):
        subpanel = '_'.join([str(self.page_id), str(self.panel_id)])
        img_path = os.path.join(panel_dir, str(self.book_id), f'{subpanel}.jpg')
        return img_path

    def generate_img_embedding_path(self, embedding_dir: str, extension: str = 'pkl'):
        subpanel = '_'.join([str(self.page_id), str(self.panel_id)])
        embedding_path = os.path.join(embedding_dir, str(self.book_id), f'{subpanel}.{extension}')
        return embedding_path

    def get_speech_bubble_bounding_boxes(self) -> List[Any]:
        speech_bubble_bbs = list(
            map(lambda text: [text.x1, text.y1, text.x2, text.y2], self.texts if self.texts is not None else []))
        speech_bubble_bbs = list(filter(lambda bb: None not in bb, speech_bubble_bbs))
        return speech_bubble_bbs
