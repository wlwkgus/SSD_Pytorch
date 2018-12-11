import os
from PIL import Image
from torch.utils.data import Dataset
import pickle
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
import cv2
import operator
import pandas as pd
import json
from tqdm import tqdm
import torch

from data.bucket_eval import bucket_eval
from data.voc_eval import voc_eval
from utils.bucket_hierarchical import StoreHierarchicalManager
from utils import label


class LabelManager(object):
    LABEL = label.LABEL

    def get_leaf_values(self, obj):
        if type(obj) == list:
            return obj
        ret = list()
        for key in obj.keys():
            ret += self.get_leaf_values(obj[key])
        return ret

    def __init__(self):
        self.region_keyword_list = self.LABEL['area']['structure'] + self.LABEL['area']['material'] + \
                                   self.LABEL['area']['pattern'] + self.get_leaf_values(self.LABEL['area']['store'])

    #         print(self.region_keyword_list)

    def region_keyword_index_to_category(self, index):
        raise NotImplemented

    def generate_label(self, row):
        region_keywords = json.loads(row.region_keywords)
        keywords = json.loads(row.keywords)
        toggle_dict = dict()
        region_dict = dict()
        region_dict['cnt'] = len(region_keywords)
        processed_region_keywords = list()
        for region in region_keywords:
            region['label'] = self.region_keyword_list.index(region['keyword'])
            processed_region_keywords.append(
                region
            )
        region_dict['areas'] = processed_region_keywords
        for toggle, item_list in self.LABEL['toggle'].items():
            toggle_list = list()
            for item in item_list:
                if item in keywords:
                    toggle_list.append(1)
                else:
                    toggle_list.append(0)
            toggle_dict[toggle] = np.asarray(toggle_list)
        return region_dict, toggle_dict


class BucketDataset(Dataset):
    def __init__(self, root_dir, name, transform=None):
        self.root_dir = root_dir
        self.name = name
        self.transform = transform
        self.label_manager = LabelManager()
        self.hierarchical_manager = StoreHierarchicalManager()
        self.df = pd.read_csv(os.path.join(self.root_dir, 'labeled_data_belongs_20181114.csv'))
        self.df = self.df[self.df.belongs != 7]
        self.train_df = self.df.sample(frac=0.80, random_state=253)
        self.rest_df = self.df[~self.df.id.isin(self.train_df.id)]
        self.validation_df = self.rest_df.sample(frac=0.50, random_state=253)
        self.test_df = self.rest_df[~self.rest_df.id.isin(self.validation_df.id)]
        if self.name == 'train':
            self.target_df = self.train_df
        elif self.name == 'validation':
            self.target_df = self.validation_df
        else:
            self.target_df = self.test_df

    def evaluate_detections(self, all_boxes, output_dir=None):
        """
        all_boxes is a list of length number-of-classes.
        Each list element is a list of length number-of-images.
        Each of those list elements is either an empty list []
        or a numpy array of detection.

        all_boxes[class][image] = [] or np.array of shape #dets x 5
        """
        self._write_bucket_results_file(all_boxes)
        self._do_python_eval(output_dir)

    def _write_bucket_results_file(self, all_boxes):
        for cls_ind, cls in enumerate(self.label_manager.region_keyword_list):
            print('Writing {} Bucket results file'.format(cls))
            filename = self._get_bucket_results_file_template().format(cls.replace('/', '_'))
            # print(filename)
            with open(filename, 'wt') as f:
                for im_ind, row in enumerate(self.target_df.itertuples()):
                    index = row.id
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    for k in range(dets.shape[0]):
                        f.write(
                            '{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.format(
                                str(index), dets[k, -1], dets[k, 0] + 1,
                                dets[k, 1] + 1, dets[k, 2] + 1,
                                dets[k, 3] + 1))

    def _do_python_eval(self, output_dir='output'):
        rootpath = os.path.join(self.root_dir, 'Bucket')
        name = 'Bucket'
        annopath = os.path.join(rootpath, 'Annotations', '{:s}.xml')
        imagesetfile = os.path.join(rootpath, 'ImageSets', 'Main',
                                    name + '.txt')
        cachedir = os.path.join(self.root_dir, 'annotations_cache')
        aps = []
        # The PASCAL VOC metric changed in 2010

        print("Generating eval dataset...")
        data_list = list()
        for i, (_, areas, info) in enumerate(tqdm(self)):
            ref_row = self.target_df.iloc[i]
            row = dict()
            row['id'] = ref_row.id
            row['areas'] = areas
            row['info'] = info
            data_list.append(row)

        use_07_metric = True
        print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
        if output_dir is not None and not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        ap_dict = dict()
        for i, cls in enumerate(self.label_manager.region_keyword_list):
            filename = self._get_bucket_results_file_template().format(cls.replace('/', '_'))
            rec, prec, ap = bucket_eval(
                filename,
                data_list,
                self.label_manager,
                cls,
                ovthresh=0.5,
                use_07_metric=use_07_metric)
            if rec is None:
                continue
            aps += [ap]
            print('AP for {} = {:.4f}'.format(cls, ap))
            ap_dict[cls] = ap
            if output_dir is not None:
                with open(os.path.join(output_dir, cls.replace('/', '_') + '_pr.pkl'),
                          'wb') as f:
                    pickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
        print('Mean AP = {:.4f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('Results:')
        for ap in aps:
            print('{:.3f}'.format(ap))
        print('{:.3f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('')
        print('--------------------------------------------------------------')
        print('Results computed with the **unofficial** Python eval code.')
        print('Results should be very close to the official MATLAB eval code.')
        print('Recompute with `./tools/reval.py --matlab ...` for your paper.')
        print('-- Thanks, The Management')
        print('--------------------------------------------------------------')

        sorted_ap_dict_items = sorted(ap_dict.items(), key=operator.itemgetter(1), reverse=True)
        print("[ Top 30 ]")

        for tup in sorted_ap_dict_items[:30]:
            print("{} : {}".format(tup[0], tup[1]))

        top_array = np.asarray([tup[1] for tup in sorted_ap_dict_items[:30]])
        print(">>> Top 30 mean AP : {}".format(top_array.mean()))
        print(">>> Top 30 stddev AP : {}".format(top_array.std()))

        sorted_ap_dict_items = sorted(ap_dict.items(), key=operator.itemgetter(1))
        print("[ Worst 30 ]")

        for tup in sorted_ap_dict_items[:30]:
            print("{} : {}".format(tup[0], tup[1]))

        top_array = np.asarray([tup[1] for tup in sorted_ap_dict_items[:30]])
        print(">>> Worst 30 mean AP : {}".format(top_array.mean()))
        print(">>> Worst 30 stddev AP : {}".format(top_array.std()))

    def _get_bucket_results_file_template(self):
        filename = 'comp3_det_test' + '_{:s}.txt'
        filedir = os.path.join(self.root_dir, 'results', 'Bucket', 'Main')
        if not os.path.exists(filedir):
            os.makedirs(filedir)
        path = os.path.join(filedir, filename)
        return path

    def __len__(self):
        return len(self.target_df)

    def __getitem__(self, idx):
        row = self.target_df.iloc[idx]
        region_dict, toggle_dict = self.label_manager.generate_label(row)
        image_dir = os.path.join(self.root_dir, 'imgs')
        image_dir = os.path.join(image_dir, '{}.jpg'.format(row.id))
        img = cv2.imread(image_dir, cv2.IMREAD_COLOR)

        # No toggle label for object detection.
        # sample['color'] = toggle_dict['color']
        # sample['structure'] = toggle_dict['structure']
        # sample['material'] = toggle_dict['material']
        # sample['pattern'] = toggle_dict['pattern']
        # sample['manner'] = toggle_dict['manner']

        # Region Label
        areas = list()
        height, width, _ = img.shape
        info = [width, height]
        for i, region in enumerate(region_dict['areas']):
            anno = np.concatenate(
                (
                    region['coordinates'],
                    np.zeros(1)
                )
            )
            anno[4] = region['label']
            anno[[0, 2]] *= width / 100
            anno[[1, 3]] *= height / 100
            anno = np.floor(anno)
            areas.append(
                anno
            )
        areas = np.asarray(areas).reshape(-1, 5)

        if self.name != 'test':
            if self.transform is not None:
                img, areas = self.transform(img, areas)
        else:
            if self.transform is not None:
                img = self.transform(img)
        return img, areas, info, three_depth_label


def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on 0 dim
    """
    targets = []
    imgs = []
    img_info = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))
        img_info.append(torch.FloatTensor(sample[2]))
    return torch.stack(imgs, 0), targets, img_info
