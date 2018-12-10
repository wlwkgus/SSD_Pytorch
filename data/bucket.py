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


class LabelManager(object):
    LABEL = {
        "area": {
            "structure": ["계단", "창문/샷시(외창)", "유리파티션", "간접조명(시사시조명)"],
            "material": ["모자이크 타일", "직사각형 타일", "직사각형 타일(지그재그)", "정사각형 타일", "정사각형 타일(지그재그)", "육각타일", "헤링본타일", "헤링본마루",
                         "대리석타일", "데크타일", "파벽돌 (타일)", "주방 상부장", "주방 하부장", "세면대", "변기", "욕실수납장", "욕조", "폴딩도어", "중문",
                         "일반 도어(방문)", "1단 벽선반", "2단 벽선반", "3단 벽선반", "4단 이상 벽선반", "웨인스코팅", "네온사인"],
            "pattern": ["헤링본", "체크", "스트라이프", "우드(나무무늬)", "마블 (대리석무늬)", "라탄", "페르시안", "레이스", "꽃/식물무늬", "도트(땡땡이)"],
            "store": {
                "가구": {
                    "소파/거실가구": [
                        "가죽소파 1인",
                        "패브릭소파 1인",
                        "리클라이너소파 1인",
                        "가죽소파 2인 이상",
                        "패브릭소파 2인 이상",
                        "리클라이너소파 2인 이상",
                        "빈백소파",
                        "좌식소파",
                        "소파베드",
                        "소파스툴",
                        "거실장/TV장",
                        "거실/소파 테이블",
                        "진열/장식장"
                    ],
                    "침실가구": [
                        "침대",
                        "매트리스",
                        "화장대",
                        "옷장/붙박이장",
                        "드레스룸",
                        "거울",
                        "서랍장",
                        "콘솔",
                        "협탁",
                        "침대",
                        "이층 침대"
                    ],
                    "주방가구": [
                        "2인 식탁/세트",
                        "4인 식탁/세트",
                        "6인 식탁/세트",
                        "홈바",
                        "아일랜드 식탁",
                        "레인지대",
                        "주방수납장/그릇장",
                        "식탁의자"
                    ],
                    "수납가구": [
                        "수납장",
                        "철제수납장",
                        "행거",
                        "선반",
                        "공간박스",
                        "현관수납/신발장",
                        "코너장",
                        "이동식수납/트롤리",
                    ],
                    "학생/서재가구": [
                        "책장",
                        "책상",
                        "오피스/학생 의자",
                        "오피스서랍장"
                    ],
                    "의자/스툴": [
                        "패브릭/가죽 체어",
                        "플라스틱 체어",
                        "철제 체어",
                        "우드 체어",
                        "바 체어",
                        "오피스 체어",
                        "스툴",
                        "좌식의자",
                        "벤치",
                        "안락의자/흔들의자",
                        "의자 발받침"
                    ],
                    "테이블": [
                        "좌식테이블",
                        "사이드테이블",
                        "접이식테이블"
                    ],
                    "유아동가구": [
                        "책상/테이블",
                        "책장/책꽂이",
                        "소파/빈백",
                        "의자",
                        "서랍장/옷장",
                        "침대/벙커침대",
                        "기타"
                    ],
                },
                "패브릭": {
                    "침구": [
                        "이불커버",
                        "베개커버",
                        "요/패드/침대 커버"
                    ],
                    "커튼/블라인드": [
                        "커튼",
                        "바란스/가리개",
                        "블라인드",
                        "롤스크린",
                        "캐노피"
                    ],
                    "카페트/러그": [
                        "러그/카페트",
                        "발매트/주방매트",
                        "놀이방/안전매트",
                    ],
                    "쿠션/방석/담요": [
                        "쿠션",
                        "방석",
                        "대방석",
                        "기능성/바디필로우 외",
                        "담요/스로우 외"
                    ]
                },
                "홈데코/조명": {
                    "조명": [
                        "볼 조명/앵두 전구",
                        "천장조명",
                        "팬던트조명",
                        "레일조명",
                        "플로어조명",
                        "테이블 조명",
                        "데스크 스탠드",
                        "벽조명",
                        "레터링 조명"
                    ],
                    "시계": [
                        "벽시계",
                        "알람/탁상시계",
                        "스탠드시계"
                    ],
                    "플라워/식물": [
                        "식물",
                        "꽃(생화/조화)",
                        "화병",
                        "화분/화분 커버",
                        "리스/가랜드"
                    ],
                    "액자/월데코": [
                        "탁상 액자",
                        "벽걸이 액자",
                        "패브릭 포스터",
                        "디자인/사진 포스터",
                        "마크라메"
                    ],
                    "캔들/디퓨저": [
                        "캔들",
                        "디퓨져",
                        "캔들 홀더/워머"
                    ],
                    "장식소품": [
                        "모빌",
                        "가랜드"
                    ],
                    "데스크/디자인 문구": [
                        "달력",
                        "카드",
                        "포스터",
                        "타공판/보드"
                    ],
                    "크리스마스": [
                        "트리"
                    ]
                },
                "수납/생활": {
                    "수납/바스켓": [
                        "리빙박스/수납함",
                        "바구니/바스켓",
                        "데스크/화장대수납",
                        "이동식정리함",
                        "기타정리/수납용품"
                    ],
                    "선반/진열대": [
                        "벽걸이 수납선반",
                        "신발정리대",
                        "우산꽂이"
                    ],
                    "행거/옷걸이": [
                        "스탠드행거",
                        "설치형행거",
                        "벽걸이행거",
                        "폴행거/봉행거"
                    ],
                    "욕실용품": [
                        "욕실수납",
                        "샤워기/수전용품",
                        "욕실/발 매트",
                        "샤워 커튼/봉",
                        "기타"
                    ],
                    "청소용품": [
                        "휴지통",
                        "분리수거함"
                    ],
                    "세탁용품": [
                        "빨래건조대",
                        "빨래바구니/보관함"
                    ],
                    "계절용품": [
                        "난방텐트"
                    ]
                },
                "가전": {
                    "계절가전": [
                        "선풍기",
                        "에어컨",
                        "히터"
                    ],
                    "음향/영상가전": [
                        "프로젝터",
                        "TV / 모니터",
                        "스피커",
                        "턴테이블"
                    ],
                    "주방가전": [
                        "블랜더/믹서",
                        "토스터/홈베이킹",
                        "전기주전자",
                        "커피 메이커/머신",
                        "밥솥/찜기",
                        "오븐/전자레인지",
                        "전기그릴",
                        "냉장고",
                        "가스레인지/인덕션",
                        "기타"
                    ],
                }
            }
        },
        "toggle": {
            "color": ["블랙", "화이트", "그레이", "베이지", "라이트 브라운", "브라운", "레드", "핑크", "옐로우", "그린", "민트", "블루", "네이비"],
            "structure": ["파우더룸", "복층", "이층집", "옥상", "마당", "가벽/파티션", "노출 천장", "사선형 천장", "우물천장", "ㄷ자주방", "ㄱ자주방", "ㅡ자주방",
                          "二 자 주방", "아일랜드 시공", "대면형 주방", "복도식 구조", "아치형 통로", "붙박이장 및 제작가구"],
            "material": ["유광 바닥타일 ", "무광 바닥타일", "유광 벽타일 ", "무광 벽타일", "노출 콘크리트", "기타 패턴타일", "데크", "싱크대", "원목 상판",
                         "(인조)대리석 상판", "철제상판", "골드수전", "블랙수전", "실버수전", "수도수전", "샤워수전", "해바라기 샤워기", "샤워부스", "여닫이중문",
                         "2연동 중문", "3연동 중문", "슬라이딩 중문 도어", "ㄱ자 중문", "슬라이딩 도어(중문X)", "터닝도어", "루버셔터", "망입유리", "아트월"],
            "pattern": ["메탈(철제)", "(투명/불투명)유리", "투톤 컬러벽"],
            "manner": ["자연광", "인공조명", "흰색조명", "노란조명", "간접조명", "외부전망", "밝음", "어두움", "피규어"]
        }
    }

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
        return img, areas, info


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
