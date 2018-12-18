# -*- coding: utf-8 -*-
# Written by yq_yao

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from utils.box_utils import match, log_sum_exp, refine_match
from layers.modules import WeightSoftmaxLoss, WeightSmoothL1Loss
from utils.depth_manager import DepthManager

GPU = False
if torch.cuda.is_available():
    GPU = True
    torch.set_default_tensor_type('torch.cuda.FloatTensor')


class RefineMultiBoxLoss(nn.Module):
    """SSD Weighted Loss Function
    Compute Targets:
        1) Produce Confidence Target Indices by matching  ground truth boxes
           with (default) 'priorboxes' that have jaccard index > threshold parameter
           (default threshold: 0.5).
        2) Produce localization target by 'encoding' variance into offsets of ground
           truth boxes and their matched  'priorboxes'.
        3) Hard negative mining to filter the excessive number of negative examples
           that comes with using a large number of default bounding boxes.
           (default negative:positive ratio 3:1)
    Objective Loss:
        L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
        weighted by α which is set to 1 by cross val.
        Args:
            c: class confidences,
            l: predicted boxes,
            g: ground truth boxes
            N: number of matched default boxes
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    """

    def __init__(self, cfg, num_classes, hierarchical_num_classes=None):
        super(RefineMultiBoxLoss, self).__init__()
        self.cfg = cfg
        self.size = cfg.MODEL.SIZE
        if self.size == '300':
            size_cfg = cfg.SMALL
        else:
            size_cfg = cfg.BIG
        self.variance = size_cfg.VARIANCE
        self.num_classes = num_classes
        self.threshold = cfg.TRAIN.OVERLAP
        self.OHEM = cfg.TRAIN.OHEM
        self.negpos_ratio = cfg.TRAIN.NEG_RATIO
        self.object_score = cfg.MODEL.OBJECT_SCORE
        self.variance = size_cfg.VARIANCE
        if cfg.TRAIN.FOCAL_LOSS:
            if cfg.TRAIN.FOCAL_LOSS_TYPE == 'SOFTMAX':
                self.focaloss = FocalLossSoftmax(
                    self.num_classes, gamma=2, size_average=False)
            else:
                self.focaloss = FocalLossSigmoid()
        self.hierarchical_num_classes = None
        if hierarchical_num_classes is not None:
            if type(hierarchical_num_classes) == list:
                numsum = 0
                for num in hierarchical_num_classes:
                    numsum += num
                if numsum != num_classes:
                    raise Exception("numsum should be same with num classes!")
                self.hierarchical_num_classes = hierarchical_num_classes
            else:
                raise Exception("hierarchical num classes shuold be a type of list.")

    def forward(self,
                predictions,
                targets,
                use_arm=False,
                filter_object=False,
                debug=False):
        """Multibox Loss
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds,
            and prior boxes from SSD net.
                conf shape: torch.size(batch_size,num_priors,num_classes)
                loc shape: torch.size(batch_size,num_priors,4)
                priors shape: torch.size(num_priors,4)

            ground_truth (tensor): Ground truth boxes and labels for a batch,
                shape: [batch_size,num_objs,5] (last idx is the label).
        """
        # arm_loc_data, arm_conf_data, loc_data, conf_data, priors = predictions
        if use_arm:
            arm_loc_data, arm_conf_data, loc_data, conf_data, priors = predictions
        else:
            loc_data, conf_data, _, _, priors = predictions
        num = loc_data.size(0)
        priors = priors[:loc_data.size(1), :]
        num_priors = (priors.size(0))
        num_classes = self.num_classes

        # match priors (default boxes) and ground truth boxes
        if self.hierarchical_num_classes is None:
            loc_t = torch.Tensor(num, num_priors, 4)
            conf_t = torch.LongTensor(num, num_priors)
            defaults = priors.data
            for idx in range(num):
                truths = targets[idx][:, :-1].data
                labels = targets[idx][:, -1].data
                if self.num_classes == 2:
                    labels = labels > 0
                if use_arm:
                    # TODO : 분기점 여기서! hierarchy 갯수만큼 conf_t를 만들어야 함.
                    # TODO : 주어진 라벨 값을 토대로 로스 값에 반영하기.
                    # TODO : Should Read Labeling Code here
                    bbox_weight = refine_match(
                        self.threshold,
                        truths,
                        defaults,
                        self.variance,
                        labels,
                        loc_t,
                        conf_t,
                        idx,
                        arm_loc_data[idx].data,
                        use_weight=False)
                else:
                    match(self.threshold, truths, defaults, self.variance, labels,
                          loc_t, conf_t, idx)

            loc_t = loc_t.cuda()
            conf_t = conf_t.cuda()
            # wrap targets
            loc_t = Variable(loc_t, requires_grad=False)
            conf_t = Variable(conf_t, requires_grad=False)
        else:
            conf_ts = list()
            conf_chunk_idx_ts = list()
            # TODO : manipulate each conf_t based on depth.
            for i in range(int((targets.size(2) - 5) / 2)):
                # Filter -1 values in target info.
                conf_t = torch.LongTensor(num, num_priors)
                conf_chunk_idx_t = torch.LongTensor(num, num_priors)
                conf_ts.append(
                    conf_t
                )
                conf_chunk_idx_ts.append(
                    conf_chunk_idx_t
                )
            loc_t = torch.Tensor(num, num_priors, 4)
            defaults = priors.data

            if use_arm:
                for idx in range(num):
                    truths = targets[idx][:, :4].data
                    # TODO : 여기 옵션 조정하기!!
                    depth_labels = targets[idx][:, [5, 7, 9, 10]].data
                    all_chunk_idxes = targets[idx][:, [6, 8, 10, 12]].data
                    for i in range(depth_labels.size(1)):
                        labels = depth_labels[:, i]
                        labels = labels[labels[:, 0] >= 0, :]
                        chunk_idxes = all_chunk_idxes[:, i]
                        chunk_idxes = chunk_idxes[chunk_idxes[:, 0] >= 0, :]
                        bbox_weight = refine_match(
                            self.threshold,
                            truths,
                            defaults,
                            self.variance,
                            labels,  # 0은 백그라운드로 처리해버림.
                            loc_t,
                            conf_ts[i],
                            idx,
                            arm_loc_data[idx].data,
                            assign_loc=i < 1 and idx < 1,
                            assign_background_value=0 if i == 0 else -1,
                            use_weight=False,
                            is_chunk_idx=True,
                            chunk_idxes=chunk_idxes,
                            conf_chunk_idx_t=conf_chunk_idx_ts[i],
                        )

                for i in range(len(conf_ts)):
                    conf_ts[i] = conf_ts[i].cuda()
                loc_t = loc_t.cuda()
                # wrap targets
                for i in range(len(conf_ts)):
                    conf_ts[i] = Variable(conf_ts[i], requires_grad=False)
                loc_t = Variable(loc_t, requires_grad=False)
            else:
                raise Exception("hierarchical & !use_arm is not supported!")

        if use_arm and filter_object:
            P = F.softmax(arm_conf_data, 2)
            arm_conf_data_temp = P[:, :, 1]
            object_score_index = arm_conf_data_temp <= self.object_score
            pos = conf_t > 0
            pos[object_score_index.detach()] = 0
        else:
            pos = conf_t > 0
        num_pos = pos.sum(1, keepdim=True)
        if debug:
            if use_arm:
                print("odm pos num: ", str(loc_t.size(0)), str(loc_t.size(1)))
            else:
                print("arm pos num", str(loc_t.size(0)), str(loc_t.size(1)))

        if self.OHEM:
            # Compute max conf across batch for hard negative mining
            batch_conf = conf_data.view(-1, self.num_classes)
            if self.hierarchical_num_classes is None:
                loss_c = log_sum_exp(batch_conf) - batch_conf.gather(
                    1, conf_t.view(-1, 1))

                # Hard Negative Mining
                # TODO : Need to see value of conf_t.
                loss_c[pos.view(-1, 1)] = 0  # filter out pos boxes for now
                loss_c = loss_c.view(num, -1)
                _, loss_idx = loss_c.sort(1, descending=True)
                _, idx_rank = loss_idx.sort(1)
                num_pos = pos.long().sum(1, keepdim=True)
                num_neg = torch.clamp(
                    self.negpos_ratio * num_pos, max=pos.size(1) - 1)
                neg = idx_rank < num_neg.expand_as(idx_rank)

                # Confidence Loss Including Positive and Negative Examples
                pos_idx = pos.unsqueeze(2).expand_as(conf_data)
                neg_idx = neg.unsqueeze(2).expand_as(conf_data)

                conf_p = conf_data[(pos_idx + neg_idx).gt(0)].view(
                    -1, self.num_classes)

                targets_weighted = conf_t[(pos + neg).gt(0)]
                try:
                    loss_c = F.cross_entropy(
                        conf_p, targets_weighted, size_average=False)
                except:
                    print('targets')
                    print(targets)
                    print('loc_data')
                    print(loc_data)
                    print('conf_t')
                    print(conf_t)
                    print('neg')
                    print(neg)
                    print('conf_p')
                    print(conf_p)
                    print('targets_weighted')
                    print(targets_weighted)
                    print('neg')
                    print(neg)
                    print('num_neg')
                    print(num_neg)
                    print('num_pos')
                    print(num_pos)
                    raise
            else:
                batch_confs = list()
                cumsum = 0
                for num_classes in self.hierarchical_num_classes:
                    batch_confs.append(
                        batch_conf[..., cumsum:cumsum+num_classes]
                    )
                    cumsum += num_classes

                # TODO : conf_t 계층에 맞춰서 모양 맞게 필터링 된 상태,
                loss_c_sum = 0.

                for i in range(4):
                    # batch_confs : list_of_num_classes 에 따라서 쪼개진 상태
                    # conf_ts : 4계층에 따라서 쪼개진 상태.
                    if i > 0:
                        loss_c = 0.
                        conf_chunk_idx_t = conf_chunk_idx_ts[i]
                        conf_t = conf_ts[i]
                        for batch in range(num):

                            log_sum_exp(batch_confs[batch][conf_chunk_idx_t])
                            pass

                        # TODO : here
                        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(
                            1, conf_t.view(-1, 1))

                        # Hard Negative Mining
                        # TODO : Need to see value of conf_t.
                        loss_c[pos.view(-1, 1)] = 0  # filter out pos boxes for now
                        loss_c = loss_c.view(num, -1)
                        _, loss_idx = loss_c.sort(1, descending=True)
                        _, idx_rank = loss_idx.sort(1)
                        num_pos = pos.long().sum(1, keepdim=True)
                        num_neg = torch.clamp(
                            self.negpos_ratio * num_pos, max=pos.size(1) - 1)
                        neg = idx_rank < num_neg.expand_as(idx_rank)

                        # Confidence Loss Including Positive and Negative Examples
                        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
                        neg_idx = neg.unsqueeze(2).expand_as(conf_data)

                        conf_p = conf_data[(pos_idx + neg_idx).gt(0)].view(
                            -1, self.num_classes)

                        targets_weighted = conf_t[(pos + neg).gt(0)]
                        loss_c = F.cross_entropy(
                            conf_p, targets_weighted, size_average=False)
                        loss_c = loss_c_sum / len(depths)
                    else:
                        # TODO : fix here also
                        loss_c = 0.
                        conf_chunk_idx_t = conf_chunk_idx_ts[i]
                        conf_t = conf_ts[i]
                        conf_t = conf_t[...,]
                        for batch in range(num):

                            loss_c += F.cross_entropy(
                                batch_confs, conf_t, size_average=False
                            )

        else:
            loss_c = F.cross_entropy(conf_p, conf_t, size_average=False)

        # Localization Loss (Smooth L1)
        # Shape: [batch,num_priors,4]
        if num_pos.data.sum() > 0:
            pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
            loc_p = loc_data[pos_idx].view(-1, 4)
            loc_t = loc_t[pos_idx].view(-1, 4)
            loss_l = F.smooth_l1_loss(loc_p, loc_t, size_average=False)
            N = num_pos.data.sum()
        else:
            loss_l = 0
            N = 1.0

        loss_l /= float(N)
        loss_c /= float(N)
        return loss_l, loss_c
