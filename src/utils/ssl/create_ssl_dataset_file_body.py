import sys
from typing import List

from tqdm import tqdm
from src.utils.basic_utils import search_files
import os
import pandas as pd
from shapely.geometry import box
import glob


def box_intersection_rate(source_bb: List, target_bb: List) -> float:
    box_1, box_2 = box(*source_bb), box(*target_bb)
    if box_1.area == 0:
        return 0
    return box_1.intersection(box_2).area / box_1.area


invalidation_width_count = 0
invalidation_height_count = 0
invalidation_score_count = 0
invalidation_intersection_count = 0
invalidation_row_not_found = 0


def _validate_item(item) -> bool:
    global invalidation_width_count
    global invalidation_height_count
    global invalidation_score_count
    global invalidation_intersection_count
    global invalidation_row_not_found
    # DASS_Det_Inference/comics_crops/0/0_0/faces/0.jpg
    if 'checkpoint' in item:
        return False
    panel_path = "/".join(item.split('/')[:-2])
    item_idx = int(item.split('/')[-1].replace('.jpg', ''))
    csv_path = os.path.join(panel_path, 'face_body.csv')
    panel_df_records = pd.read_csv(csv_path).to_dict('records')
    item_row = None
    other_rows = []
    for row in panel_df_records:
        if row['index'] == item_idx and row['type'] == item_type:
            item_row = row
        elif row['type'] == item_type:
            other_rows.append(row)

    if item_row is None:
        invalidation_row_not_found += 1
        return False
    # filter by prediction score
    if float(item_row['score']) < 0.95:
        invalidation_score_count += 1
        return False
    # filter by width & height
    item_width = abs(item_row['x_0'] - item_row['x1'])
    item_height = abs(item_row['y_0'] - item_row['y_1'])
    if item_width < 32:
        invalidation_width_count += 1
        return False
    if item_height < 32:
        invalidation_height_count += 1
        return False
    # filter by intersection rate with others
    for other_row in other_rows:
        source_bb = [item_row['x_0'], item_row['y_0'], item_row['x1'], item_row['y_1']]
        target_bb = [other_row['x_0'], other_row['y_0'], other_row['x1'], other_row['y_1']]
        intersection_rate = box_intersection_rate(source_bb, target_bb)
        if intersection_rate > 0.20:
            invalidation_intersection_count += 1
            return False

    return True


def save_all_dataset():
    dataset = search_files('.jpg',
                           comics_crops_dir,
                           filename_condition=lambda filename: 'bodies' in filename,
                           limit=limit_search_files,
                           enable_tqdm=True)
    all_item_paths_df = pd.DataFrame(dataset, columns=['img_path'])
    all_item_paths_df.to_csv(
        dataset_path,
        encoding='utf-8',
        index=False)


def load_paths(min_comics_series: int = None,
               max_comics_series: int = None) -> List[str]:
    items = pd.read_csv(dataset_path)['img_path'].tolist()

    if min_comics_series is not None and max_comics_series is not None:
        def filter_func(x):
            series_id = int(x.split('/')[-4])
            return min_comics_series <= series_id and series_id < max_comics_series

        items = list(filter(filter_func, items))
    return items


item_type = 'body'
comics_crops_dir = '/scratch/users/gsoykan20/projects/DASS_Det_Inference/comics_crops/'
limit_search_files = None
dataset_path = f'/scratch/users/gsoykan20/projects/amazing-mysteries-of-gutter-demystified/src/utils/ssl/comics_crops_{item_type}_paths.csv'


def create_filtered_csv():
    if not os.path.exists(dataset_path):
        save_all_dataset()

    min_comics_series = None
    max_comics_series = None
    if len(sys.argv) >= 2:
        min_comics_series = int(sys.argv[1])
        max_comics_series = int(sys.argv[2])
    all_paths = load_paths(min_comics_series, max_comics_series)
    print('initial dataset length: ', len(all_paths))
    all_paths = [x for x in tqdm(all_paths, f'filtering {item_type} dataset...') if _validate_item(x)]
    print('filtered dataset length: ', len(all_paths))
    print('invalidation counts...')
    print(invalidation_row_not_found, invalidation_score_count, invalidation_width_count, invalidation_height_count,
          invalidation_intersection_count)
    all_item_paths_df = pd.DataFrame(all_paths, columns=['img_path'])
    all_item_paths_df.to_csv(
        f'filtered_{min_comics_series}-{max_comics_series}.csv',
        encoding='utf-8',
        index=False)


def merge_filtered_csvs():
    csv_files = glob.glob('*.{}'.format('csv'))
    csv_files = list(filter(lambda x: 'filtered_' in x, csv_files))
    breakpoint()
    df_concat = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
    df_concat.to_csv(
        f'filtered_all_{item_type}.csv',
        encoding='utf-8',
        index=False)


def clean_filtered_csv():
    filtered_all_csv_path = '/scratch/users/gsoykan20/projects/amazing-mysteries-of-gutter-demystified/data/ssl/filtered_all_body.csv'
    df_concat = pd.read_csv(filtered_all_csv_path)
    df_concat = df_concat[~df_concat['img_path'].str.contains('faces')]
    df_concat.to_csv(
        f'filtered_all_{item_type}.csv',
        encoding='utf-8',
        index=False)


if __name__ == '__main__':
    # create_filtered_csv()
    # merge_filtered_csvs()
    clean_filtered_csv()
