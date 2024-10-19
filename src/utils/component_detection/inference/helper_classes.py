import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from itertools import groupby
from typing import Optional, List, Tuple, Union

import cv2
import numpy as np
import pandas as pd
import torch
from torch import Tensor

from src.models.pml_id_net_fine_tuned_ssl_backbone_face_body_module import PMLIdNetFineTunedSSLBackboneFaceBodyLitModule

from src.utils.basic_utils import cat_df

from src.utils.character_identity.identity_assignment.character_identity_assigner import CharacterIdentityAssigner


class ReadImageError(Exception):
    pass


class NoDetectionsError(Exception):
    pass


class MTLBox(ABC):
    def __init__(self,
                 box: Union[Tensor, np.ndarray]):
        if torch.is_tensor(box):
            self.box = box.cpu().numpy()
        elif isinstance(box, np.ndarray):
            self.box = box
        else:
            raise Exception('box can not be processed', box)


class MTLBoxWithId(MTLBox, ABC):
    def __init__(self,
                 box: Tensor,
                 box_id: int):
        super(MTLBoxWithId, self).__init__(box, )
        self.box_id = box_id

    @abstractmethod
    def to_csv(self,
               page_img_path: str,
               save_root_folder: str,
               original_img_shape: Tuple[int, int, int],  # h, w, c
               transformed_img_shape: Tuple[int, int, int]) -> pd.DataFrame:
        ...


class MTLBoxInPanel(MTLBoxWithId, ABC):
    def __init__(self,
                 box: Tensor,
                 box_id: int,
                 panel_id: Optional[int]):
        super(MTLBoxInPanel, self).__init__(box, box_id)
        self.panel_id = panel_id


@dataclass
class MTLNarrative(MTLBoxInPanel):
    def __init__(self,
                 box: Union[Tensor, np.ndarray],
                 box_id: int,
                 panel_id: Optional[int],
                 panel_order: Optional[int], ):
        super(MTLNarrative, self).__init__(box, box_id, panel_id)
        self.panel_order = panel_order

    def to_csv(self,
               page_img_path: str,
               save_root_folder: str,
               original_img_shape: Tuple[int, int, int],
               transformed_img_shape: Tuple[int, int, int]) -> pd.DataFrame:
        comic_series, page = page_img_path.split('/')[-2:]
        page = page.split('.')[0]
        scale_h_w = np.array(original_img_shape[:2]) / np.array(transformed_img_shape[:2])

        def scale_to_original_img(bb):
            x_min, y_min, x_max, y_max = bb
            x_min, x_max = int(x_min * scale_h_w[1]), int(x_max * scale_h_w[1])
            y_min, y_max = int(y_min * scale_h_w[0]), int(y_max * scale_h_w[0])
            return [x_min, y_min, x_max, y_max]

        x_0, y_0, x_1, y_1 = scale_to_original_img(self.box)

        narrative_data = {
            'series': [comic_series],
            'page': [page],
            'box_id': [self.box_id],
            'panel_id': [self.panel_id],
            'panel_order': [self.panel_order],
            'x_0': [x_0],
            'y_0': [y_0],
            'x_1': [x_1],
            'y_1': [y_1],
        }
        narrative_df = pd.DataFrame(narrative_data)
        return narrative_df


@dataclass
class MTLSpeech(MTLBoxInPanel):
    def __init__(self,
                 box: Union[Tensor, np.ndarray],
                 box_id: int,
                 panel_id: Optional[int],
                 panel_order: Optional[int],
                 order: Optional[int],
                 segm_mask: np.ndarray):
        super(MTLSpeech, self).__init__(box, box_id, panel_id)
        self.order = order
        self.panel_order = panel_order
        self.segm_mask = segm_mask

    def to_csv(self,
               page_img_path: str,
               save_root_folder: str,
               original_img_shape: Tuple[int, int, int],
               transformed_img_shape: Tuple[int, int, int]) -> pd.DataFrame:
        comic_series, page = page_img_path.split('/')[-2:]
        page = page.split('.')[0]
        scale_h_w = np.array(original_img_shape[:2]) / np.array(transformed_img_shape[:2])

        def scale_to_original_img(bb):
            x_min, y_min, x_max, y_max = bb
            x_min, x_max = int(x_min * scale_h_w[1]), int(x_max * scale_h_w[1])
            y_min, y_max = int(y_min * scale_h_w[0]), int(y_max * scale_h_w[0])
            return [x_min, y_min, x_max, y_max]

        x_0, y_0, x_1, y_1 = scale_to_original_img(self.box)
        mask_resized = cv2.resize(self.segm_mask, (original_img_shape[:2][1], original_img_shape[:2][0]))

        speech_data = {
            'series': [comic_series],
            'page': [page],
            'box_id': [self.box_id],
            'order': [self.order],
            'panel_order': [self.panel_order],
            'panel_id': [self.panel_id],
            'x_0': [x_0],
            'y_0': [y_0],
            'x_1': [x_1],
            'y_1': [y_1],
            'mask': [mask_resized]
        }
        speech_df = pd.DataFrame(speech_data)

        speech_df['mask'] = speech_df['mask'].apply(
            lambda mask: cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0].tolist())

        # Convert the polygon back to a mask
        # df['mask'] = df['mask'].apply(
        #     lambda polygon: cv2.fillPoly(np.zeros((3, 3), dtype=np.uint8), [np.array(polygon, dtype=np.int32)]))

        return speech_df


@dataclass
class MTLFace(MTLBoxInPanel):
    def __init__(self,
                 box: Union[Tensor, np.ndarray],
                 box_id: int,
                 panel_id: Optional[int]):
        super(MTLFace, self).__init__(box, box_id, panel_id)

    def to_csv(self,
               page_img_path: str,
               save_root_folder: str,
               original_img_shape: Tuple[int, int, int],
               transformed_img_shape: Tuple[int, int, int]) -> pd.DataFrame:
        comic_series, page = page_img_path.split('/')[-2:]
        page = page.split('.')[0]
        scale_h_w = np.array(original_img_shape[:2]) / np.array(transformed_img_shape[:2])

        def scale_to_original_img(bb):
            x_min, y_min, x_max, y_max = bb
            x_min, x_max = int(x_min * scale_h_w[1]), int(x_max * scale_h_w[1])
            y_min, y_max = int(y_min * scale_h_w[0]), int(y_max * scale_h_w[0])
            return [x_min, y_min, x_max, y_max]

        x_0, y_0, x_1, y_1 = scale_to_original_img(self.box)

        face_data = {
            'series': [comic_series],
            'page': [page],
            'box_id': [self.box_id],
            'panel_id': [self.panel_id],
            'x_0': [x_0],
            'y_0': [y_0],
            'x_1': [x_1],
            'y_1': [y_1],
        }
        face_df = pd.DataFrame(face_data)
        return face_df


@dataclass
class MTLBody(MTLBoxInPanel):
    def __init__(self,
                 box: Union[Tensor, np.ndarray],
                 box_id: int,
                 panel_id: Optional[int]):
        super(MTLBody, self).__init__(box, box_id, panel_id)

    def to_csv(self,
               page_img_path: str,
               save_root_folder: str,
               original_img_shape: Tuple[int, int, int],
               transformed_img_shape: Tuple[int, int, int]) -> pd.DataFrame:
        comic_series, page = page_img_path.split('/')[-2:]
        page = page.split('.')[0]
        scale_h_w = np.array(original_img_shape[:2]) / np.array(transformed_img_shape[:2])

        def scale_to_original_img(bb):
            x_min, y_min, x_max, y_max = bb
            x_min, x_max = int(x_min * scale_h_w[1]), int(x_max * scale_h_w[1])
            y_min, y_max = int(y_min * scale_h_w[0]), int(y_max * scale_h_w[0])
            return [x_min, y_min, x_max, y_max]

        x_0, y_0, x_1, y_1 = scale_to_original_img(self.box)

        body_data = {
            'series': [comic_series],
            'page': [page],
            'box_id': [self.box_id],
            'panel_id': [self.panel_id],
            'x_0': [x_0],
            'y_0': [y_0],
            'x_1': [x_1],
            'y_1': [y_1],
        }
        body_df = pd.DataFrame(body_data)
        return body_df


@dataclass
class MTLCharacter(MTLBox):
    def __init__(self,
                 box: Union[Tensor, np.ndarray],
                 panel_id: Optional[int],
                 panel_order: Optional[int],
                 face: Optional[MTLFace],
                 body: Optional[MTLBody],
                 speech_ids: List[int],
                 identity: Optional[str] = None):
        super().__init__(box)
        self.box = box.cpu().numpy() if torch.is_tensor(box) else box
        self.panel_id = panel_id
        self.panel_order = panel_order
        self.face = face
        self.body = body
        self.speech_ids = speech_ids
        self.identity = identity

    def to_csv(self,
               page_img_path: str,
               save_root_folder: str,
               original_img_shape: Tuple[int, int, int],
               transformed_img_shape: Tuple[int, int, int]) -> pd.DataFrame:
        comic_series, page = page_img_path.split('/')[-2:]
        page = page.split('.')[0]
        scale_h_w = np.array(original_img_shape[:2]) / np.array(transformed_img_shape[:2])

        def scale_to_original_img(bb):
            x_min, y_min, x_max, y_max = bb
            x_min, x_max = int(x_min * scale_h_w[1]), int(x_max * scale_h_w[1])
            y_min, y_max = int(y_min * scale_h_w[0]), int(y_max * scale_h_w[0])
            return [x_min, y_min, x_max, y_max]

        x_0, y_0, x_1, y_1 = scale_to_original_img(self.box)

        char_data = {
            'series': [comic_series],
            'page': [page],
            'panel_id': [self.panel_id],
            'panel_order': [self.panel_order],
            'x_0': [x_0],
            'y_0': [y_0],
            'x_1': [x_1],
            'y_1': [y_1],
            'speech_ids': [self.speech_ids],
            'face_box_id': [None],
            'face_x_0': [None],
            'face_y_0': [None],
            'face_x_1': [None],
            'face_y_1': [None],
            'body_box_id': [None],
            'body_x_0': [None],
            'body_y_0': [None],
            'body_x_1': [None],
            'body_y_1': [None],
        }

        if self.face is not None:
            face_box_id = self.face.box_id
            face_x_0, face_y_0, face_x_1, face_y_1 = scale_to_original_img(self.face.box)
            char_data['face_box_id'] = [face_box_id]
            char_data['face_x_0'] = [face_x_0]
            char_data['face_y_0'] = [face_y_0]
            char_data['face_x_1'] = [face_x_1]
            char_data['face_y_1'] = [face_y_1]

        if self.body is not None:
            body_box_id = self.body.box_id
            body_x_0, body_y_0, body_x_1, body_y_1 = scale_to_original_img(self.body.box)
            char_data['body_box_id'] = [body_box_id]
            char_data['body_x_0'] = [body_x_0]
            char_data['body_y_0'] = [body_y_0]
            char_data['body_x_1'] = [body_x_1]
            char_data['body_y_1'] = [body_y_1]

        char_df = pd.DataFrame(char_data)
        return char_df


@dataclass
class MTLPanel(MTLBox):
    box: np.ndarray
    box_id: int
    order: int

    narratives: List[MTLNarrative]
    speeches: List[MTLSpeech]
    characters: List[MTLCharacter]

    def all_components_to_csv(self,
                              page_img_path: str,
                              save_root_folder: str,
                              original_img_shape: Tuple[int, int, int],  # h, w, c
                              transformed_img_shape: Tuple[int, int, int]):
        comic_series, page = page_img_path.split('/')[-2:]
        page = page.split('.')[0]
        scale_h_w = np.array(original_img_shape[:2]) / np.array(transformed_img_shape[:2])

        def scale_to_original_img(bb):
            x_min, y_min, x_max, y_max = bb
            x_min, x_max = int(x_min * scale_h_w[1]), int(x_max * scale_h_w[1])
            y_min, y_max = int(y_min * scale_h_w[0]), int(y_max * scale_h_w[0])
            return [x_min, y_min, x_max, y_max]

        x_0, y_0, x_1, y_1 = scale_to_original_img(self.box)

        panel_data = {
            'series': [comic_series],
            'page': [page],
            'box_id': [self.box_id],
            'order': [self.order],
            'x_0': [x_0],
            'y_0': [y_0],
            'x_1': [x_1],
            'y_1': [y_1],
            'num_narrative': [len(self.narratives)],
            'num_speech': [len(self.speeches)],
            'num_associated_speech': [self.num_associated_speech_ids],
            'num_char': [len(self.characters)],
            'num_face': [self.num_faces],
            'num_body': [self.num_bodies],
        }
        panel_df = pd.DataFrame(panel_data)
        narrative_df = list(map(lambda x: x.to_csv(page_img_path,
                                                   save_root_folder,
                                                   original_img_shape,
                                                   transformed_img_shape), self.narratives))
        narrative_df = cat_df(narrative_df)
        speech_df = list(map(lambda x: x.to_csv(page_img_path,
                                                save_root_folder,
                                                original_img_shape,
                                                transformed_img_shape), self.speeches))
        speech_df = cat_df(speech_df)
        char_df = list(map(lambda x: x.to_csv(page_img_path,
                                              save_root_folder,
                                              original_img_shape,
                                              transformed_img_shape), self.characters))
        char_df = cat_df(char_df)

        return panel_df, narrative_df, speech_df, char_df

    @staticmethod
    def combine_all_df_components(page_img_path: str,
                                  save_root_folder: str,
                                  panels: List[pd.DataFrame],
                                  chars: List[Optional[pd.DataFrame]],
                                  narratives: List[Optional[pd.DataFrame]],
                                  speeches: List[Optional[pd.DataFrame]]):
        comic_series, page = page_img_path.split('/')[-2:]
        page = page.split('.')[0]

        # save panel
        panel_path = os.path.join(save_root_folder,
                                  str(comic_series),
                                  str(page),
                                  f'panels.csv')
        os.makedirs(os.path.dirname(panel_path), exist_ok=True)
        panel_df = cat_df(panels)
        if panel_df is not None:
            panel_df = panel_df.sort_values('order', ascending=True)
            panel_df.to_csv(panel_path, index=False)

        # save narrative
        path = os.path.join(save_root_folder,
                            str(comic_series),
                            str(page),
                            f'narratives.csv')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df = cat_df(narratives)
        if df is not None:
            df = df.sort_values('panel_order', ascending=True)
            df.to_csv(path, index=False)

        # save speech
        path = os.path.join(save_root_folder,
                            str(comic_series),
                            str(page),
                            f'speeches.csv')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df = cat_df(speeches)
        if df is not None:
            df = df.sort_values(by=['panel_order', 'order'], ascending=True)
            df.to_csv(path, index=False)

        # save char
        path = os.path.join(save_root_folder,
                            str(comic_series),
                            str(page),
                            f'chars.csv')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df = cat_df(chars)
        if df is not None:
            df = df.sort_values('panel_order', ascending=True)
            df.to_csv(path, index=False)

    def crop_all_components(self,
                            page_img_path: str,
                            save_root_folder: str,
                            original_img_shape: Tuple[int, int, int],  # h, w, c
                            transformed_img_shape: Tuple[int, int, int]):
        comic_series, page = page_img_path.split('/')[-2:]
        page = page.split('.')[0]

        scale_h_w = np.array(original_img_shape[:2]) / np.array(transformed_img_shape[:2])

        img = cv2.imread(page_img_path)

        def crop_with_mask(mask):
            mask_resized = cv2.resize(mask, (original_img_shape[:2][1], original_img_shape[:2][0]))
            contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            largest_contour = max(contours, key=cv2.contourArea)
            blank_mask = np.zeros_like(img)
            cv2.drawContours(blank_mask, contours, -1, (255, 255, 255), thickness=cv2.FILLED)
            masked_image = cv2.bitwise_and(img, blank_mask)
            x, y, w, h = cv2.boundingRect(largest_contour)
            cropped_image = masked_image[y:y + h, x:x + w]
            return cropped_image

        def crop(bb,
                 scale: Optional[float] = None,
                 make_square: bool = False):
            x_min, y_min, x_max, y_max = bb

            if scale is not None or make_square:
                center_x = (x_min + x_max) // 2
                center_y = (y_min + y_max) // 2
                width = x_max - x_min
                height = y_max - y_min

                if make_square:
                    width, height = max(width, height), max(width, height)

                if scale is not None:
                    width, height = width * scale, height * scale

                x_min, x_max = center_x - width // 2, center_x + width // 2
                y_min, y_max = center_y - height // 2, center_y + height // 2

            x_min, x_max = max(0, int(x_min * scale_h_w[1])), min(original_img_shape[1], int(x_max * scale_h_w[1]))
            y_min, y_max = max(0, int(y_min * scale_h_w[0])), min(original_img_shape[0], int(y_max * scale_h_w[0]))
            return img[y_min:y_max, x_min:x_max]

        # save panel
        panel_path = os.path.join(save_root_folder,
                                  'panel',
                                  str(comic_series),
                                  str(page),
                                  f'{self.box_id}.jpg')
        os.makedirs(os.path.dirname(panel_path), exist_ok=True)
        cv2.imwrite(panel_path, crop(self.box))

        # save narrative
        for narrative in self.narratives:
            narrative_path = os.path.join(save_root_folder,
                                          'narrative',
                                          str(comic_series),
                                          str(page),
                                          f'{narrative.box_id}.jpg')
            os.makedirs(os.path.dirname(narrative_path), exist_ok=True)
            cv2.imwrite(narrative_path, crop(narrative.box))

        # save speech-bubbles
        for speech in self.speeches:
            speech_path = os.path.join(save_root_folder,
                                       'speech',
                                       str(comic_series),
                                       str(page),
                                       f'{speech.box_id}.jpg')
            os.makedirs(os.path.dirname(speech_path), exist_ok=True)
            cv2.imwrite(speech_path, crop_with_mask(speech.segm_mask))

        # save characters (face & body)
        for character in self.characters:
            # face
            if character.face is not None:
                face_path = os.path.join(save_root_folder,
                                         'face',
                                         str(comic_series),
                                         str(page),
                                         f'{character.face.box_id}.jpg')
                os.makedirs(os.path.dirname(face_path), exist_ok=True)
                cv2.imwrite(face_path, crop(character.face.box,
                                            scale=1.2,
                                            make_square=True))
            # body
            if character.body is not None:
                body_path = os.path.join(save_root_folder,
                                         'body',
                                         str(comic_series),
                                         str(page),
                                         f'{character.body.box_id}.jpg')
                os.makedirs(os.path.dirname(body_path), exist_ok=True)
                cv2.imwrite(body_path, crop(character.body.box))

    @property
    def num_faces(self) -> int:
        num_faces = 0
        for character in self.characters:
            if character.face is not None:
                num_faces += 1
        return num_faces

    @property
    def num_bodies(self) -> int:
        num_bodies = 0
        for character in self.characters:
            if character.body is not None:
                num_bodies += 1
        return num_bodies

    @property
    def num_associated_speech_ids(self) -> int:
        num_speech_ids = 0
        for character in self.characters:
            num_speech_ids += len(character.speech_ids)
        return num_speech_ids


@dataclass
class MTLDanglingComponents:
    narratives: List[MTLNarrative]
    speeches: List[MTLSpeech]
    characters: List[MTLCharacter]


class MTLPage:
    def __init__(self,
                 series_id: int,
                 page_id: int,
                 mtl_panels: Optional[List[MTLPanel]] = None):
        self.series_id = series_id
        self.page_id = page_id
        self.mtl_panels = mtl_panels

    def reconstruct_page_from_csv(self,
                                  csv_root: str,
                                  img_root: str):
        img_path = os.path.join(img_root, str(self.series_id), f'{str(self.page_id)}.jpg')

        chars_csv_path = os.path.join(csv_root, str(self.series_id), str(self.page_id), 'chars.csv')
        narratives_csv_path = os.path.join(csv_root, str(self.series_id), str(self.page_id), 'narratives.csv')
        panels_csv_path = os.path.join(csv_root, str(self.series_id), str(self.page_id), 'panels.csv')
        speeches_csv_path = os.path.join(csv_root, str(self.series_id), str(self.page_id), 'speeches.csv')

        try:
            image = cv2.imread(img_path)
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)  # H, W, C
        except Exception as e:
            raise ReadImageError(f"An error occurred while reading the image: {e}", img_path)

        orig_h, orig_w, orig_c = image.shape

        if os.path.exists(panels_csv_path):
            panels = pd.read_csv(panels_csv_path).to_dict('records')
        else:
            panels = []

        if os.path.exists(chars_csv_path):
            chars = pd.read_csv(chars_csv_path, converters={'speech_ids': pd.eval}).to_dict('records')
        else:
            chars = []

        if os.path.exists(narratives_csv_path):
            narratives = pd.read_csv(narratives_csv_path).to_dict('records')
        else:
            narratives = []

        if os.path.exists(speeches_csv_path):
            speeches = pd.read_csv(speeches_csv_path, converters={'mask': pd.eval}).to_dict('records')
        else:
            speeches = []

        mtl_narratives = []
        mtl_speeches = []
        mtl_chars = []
        mtl_panels = []

        # handle narratives
        for row in narratives:
            box_id = int(row['box_id'])
            panel_id = int(row['panel_id'])
            panel_order = int(row['panel_order'])
            box = np.array([row['x_0'], row['y_0'], row['x_1'], row['y_1']])
            mtl_narratives.append(MTLNarrative(box, box_id, panel_id, panel_order))

        # handle speech
        for row in speeches:
            box_id = int(row['box_id'])
            box = np.array([row['x_0'], row['y_0'], row['x_1'], row['y_1']])
            panel_id = int(row['panel_id'])
            panel_order = int(row['panel_order'])
            order = int(row['order'])
            polygon = row['mask']
            segm_mask = cv2.fillPoly(np.zeros((orig_h, orig_w), dtype=np.uint8),
                                     [np.array(polygon, dtype=np.int32)],
                                     color=(255, 255, 255)) / 255
            mtl_speeches.append(MTLSpeech(box, box_id, panel_id, panel_order, order, segm_mask))

        # handle chars
        for row in chars:
            panel_id = int(row['panel_id'])
            panel_order = int(row['panel_order'])
            box = np.array([row['x_0'], row['y_0'], row['x_1'], row['y_1']])
            face_box_id = row['face_box_id']
            body_box_id = row['body_box_id']

            mtl_face = None
            if not pd.isna(face_box_id) and face_box_id != '':
                face_box = np.array([row['face_x_0'], row['face_y_0'], row['face_x_1'], row['face_y_1']])
                mtl_face = MTLFace(face_box, int(face_box_id), panel_id)

            mtl_body = None
            if not pd.isna(body_box_id) and body_box_id != '':
                body_box = np.array([row['body_x_0'], row['body_y_0'], row['body_x_1'], row['body_y_1']])
                mtl_body = MTLFace(body_box, int(body_box_id), panel_id)

            speech_ids = row['speech_ids']
            mtl_chars.append(MTLCharacter(box, panel_id, panel_order, mtl_face, mtl_body, speech_ids))

        # handle panels
        mtl_narratives.sort(key=lambda x: x.panel_id)
        mtl_speeches.sort(key=lambda x: x.panel_id)
        mtl_chars.sort(key=lambda x: x.panel_id)

        grouped_mtl_narratives = {}
        for key, group in groupby(mtl_narratives, key=lambda x: x.panel_id):
            grouped_mtl_narratives[key] = list(group)
        grouped_mtl_speeches = {}
        for key, group in groupby(mtl_speeches, key=lambda x: x.panel_id):
            grouped_mtl_speeches[key] = list(group)
        grouped_mtl_chars = {}
        for key, group in groupby(mtl_chars, key=lambda x: x.panel_id):
            grouped_mtl_chars[key] = list(group)

        for row in panels:
            box_id = int(row['box_id'])
            order = int(row['order'])
            box = np.array([row['x_0'], row['y_0'], row['x_1'], row['y_1']])
            mtl_panels.append(MTLPanel(box,
                                       box_id,
                                       order,
                                       grouped_mtl_narratives.get(box_id, []),
                                       grouped_mtl_speeches.get(box_id, []),
                                       grouped_mtl_chars.get(box_id, []), ))

        # TODO: @gsoykan - handle dangling elements...
        self.mtl_panels = mtl_panels

        return image, mtl_panels

    @staticmethod
    def visualize_page(img: Union[Tensor, np.ndarray],
                       mtl_panels: List[MTLPanel],
                       mtl_dangling_components: Optional[MTLDanglingComponents] = None):
        if torch.is_tensor(img):
            numpy_image = img.detach().cpu().numpy()
            cv2_image = np.transpose(numpy_image, (1, 2, 0))
            image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
        elif isinstance(img, np.ndarray):
            image = img

        COLORS = [
            (1, 0, 0),  # Red
            (0, 1, 0),  # Green
            (0, 0, 1),  # Blue
            (1, 1, 0),  # Yellow
            (0, 1, 1),  # Cyan
            (1, 0, 1),  # Magenta
            (1, 0.5, 0),  # Orange
            (0.5, 0, 0.5),  # Purple
            (0, 0.5, 0.5),  # Teal
            (0.8, 0.8, 0),  # Olive
            (0.5, 0.5, 0.5),  # Gray
            (0.7, 0.2, 0.3),  # Maroon
            (0.3, 0.7, 0.2),  # Lime
            (0.2, 0.3, 0.7),  # Navy
            (1, 1, 1),  # White
        ]

        def draw_bb(image,
                    box: np.ndarray,
                    label: str,
                    color,
                    put_text_below: bool = False,
                    thickness: int = 2,
                    font_scale: float = 0.8):
            cv2.rectangle(image,
                          (int(box[0]), int(box[1])),
                          (int(box[2]), int(box[3])),
                          color,
                          thickness)

            # Draw the shadow first (slightly offset from the actual text position)
            shadow_offset = 1
            shadow_color = (0.2, 0.2, 0.2)  # Shadow color (gray in this example)
            diff = -5 if not put_text_below else 5
            cv2.putText(
                image, label,
                (int(box[0] + shadow_offset), int(box[1 if not put_text_below else 3] + diff + shadow_offset)),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, shadow_color, 2)
            cv2.putText(
                image, label, (int(box[0]), int(box[1 if not put_text_below else 3] + diff)),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2)

        def draw_mask(image, mask: np.ndarray, color):
            # Convert the color values to the range [0, 255]
            color = tuple(int(c * 255) for c in color)
            red_map = np.zeros_like(mask).astype(np.uint8)
            green_map = np.zeros_like(mask).astype(np.uint8)
            blue_map = np.zeros_like(mask).astype(np.uint8)
            red_map[mask == 1], green_map[mask == 1], blue_map[mask == 1] = color
            segmentation_map = np.stack([red_map, green_map, blue_map], axis=2)
            alpha = 1
            beta = 0.002  # transparency for the segmentation map
            gamma = 0.0  # scalar added to each sum
            cv2.addWeighted(image, alpha, segmentation_map.astype(image.dtype), beta, gamma, image)

        # class labels => body, face, panel, speech-bubble, narrative
        color_counter = 0
        for mtl_panel in sorted(mtl_panels, key=lambda x: x.order, ):
            color = COLORS[0]
            draw_bb(image, mtl_panel.box, f'panel {str(mtl_panel.order)}', color)

            speech_to_color = {}
            for i, char in enumerate(mtl_panel.characters):
                if char.identity is not None:
                    color = COLORS[(int(char.identity) + 1) % len(COLORS)]
                else:
                    color = COLORS[(i + color_counter + 1) % len(COLORS)]
                if char.face is not None:
                    draw_bb(image, char.face.box, f'',
                            color, thickness=2)
                if char.body is not None:
                    draw_bb(image, char.body.box, f'',
                            color, thickness=2)
                color_counter += 1

                for speech_id in char.speech_ids:
                    associated_speeches = list(filter(lambda x: x.box_id == speech_id, mtl_panel.speeches))
                    for associated_speech in associated_speeches:
                        from_box = char.box
                        to_box = associated_speech.box

                        def box_center(box):
                            return int((box[0] + box[2]) // 2), int((box[1] + box[3]) // 2)

                        cx1, cy1 = box_center(from_box)
                        cx2, cy2 = box_center(to_box)

                        cv2.line(image, (cx1, cy1), (cx2, cy2), color, 2)

                        speech_to_color[associated_speech.box_id] = color

            for i, narrative in enumerate(mtl_panel.narratives):
                color = COLORS[(i + 4) % len(COLORS)]
                draw_bb(image, narrative.box, f'N',
                        color, put_text_below=True, )

            for i, speech in enumerate(mtl_panel.speeches):
                color = COLORS[-((i + 1) % len(COLORS))]
                color = speech_to_color.get(speech.box_id, color)
                draw_bb(image, speech.box, f'{str(speech.order)}',
                        color, put_text_below=True, thickness=1)
                mask = speech.segm_mask
                draw_mask(image, mask, color)

            cv2.imshow('Image', image)
            cv2.waitKey(0)

    @staticmethod
    def assign_identities(
            id_net: PMLIdNetFineTunedSSLBackboneFaceBodyLitModule,
            page_path: str,
            mtl_crop_root: str,
            mtl_panels: List[MTLPanel],
            mtl_dangling_components: Optional[MTLDanglingComponents] = None, ):
        comic_series, page = page_path.split('/')[-2:]
        page = page.split('.')[0]

        face_paths = []
        body_paths = []
        indices = []
        for p_i, mtl_panel in enumerate(mtl_panels):
            for c_i, char in enumerate(mtl_panel.characters):
                # root/{type}/{series}/{page}/{box_id}.jpg
                if char.face is not None:
                    face_path = os.path.join(mtl_crop_root,
                                             'face',
                                             comic_series,
                                             page,
                                             f'{str(char.face.box_id)}.jpg')
                else:
                    face_path = None

                if char.body is not None:
                    body_path = os.path.join(mtl_crop_root,
                                             'body',
                                             comic_series,
                                             page,
                                             f'{str(char.body.box_id)}.jpg')
                else:
                    body_path = None

                face_paths.append(face_path)
                body_paths.append(body_path)
                indices.append((p_i, c_i))

        embeddings = PMLIdNetFineTunedSSLBackboneFaceBodyLitModule \
            .get_embeddings_from_imgs(id_net,
                                      face_paths,
                                      body_paths).cpu().numpy()

        labels = CharacterIdentityAssigner.assign(embeddings,
                                                  algo='agglo',
                                                  linkage='average',
                                                  distance_threshold=0.77)
        for label, (p_i, c_i) in zip(labels, indices):
            mtl_panels[p_i].characters[c_i].identity = label


if __name__ == '__main__':
    # TODO: @gsoykan - handle env paths
    project_root = "/home/gsoykan/Desktop/dev/comics_ku_masters_rework/amazing-mysteries-of-gutter-demystified"
    comics_dataset_path = "/home/gsoykan/Desktop/dev/comics_ku_masters_rework/amazing-mysteries-gutter-comics"

    page = MTLPage(359, 57)
    csv_root = os.path.join(project_root, 'data/mtl_csv')
    img_root = os.path.join(comics_dataset_path, 'raw_page_images')
    image, mtl_panels = page.reconstruct_page_from_csv(csv_root, img_root)
    MTLPage.visualize_page(image / 255, mtl_panels, None)
