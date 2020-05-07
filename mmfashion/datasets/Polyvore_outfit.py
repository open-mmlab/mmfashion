import json
import os
import pickle

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset

from .registry import DATASETS


@DATASETS.register_module
class PolyvoreOutfitDataset(Dataset):
    CLASSES = None

    def __init__(self,
                 img_path,
                 annotation_path,
                 meta_file_path,
                 img_size=(224, 224),
                 text_feat_path=None,
                 text_feat_dim=6000,
                 compatibility_test_fn=None,
                 fitb_test_fn=None,
                 typespaces_fn=None,
                 train=False):

        self.img_path = img_path
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(img_size[0]),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

        # load meta data
        self.meta_data = json.load(open(meta_file_path, 'r'))

        # get item to set mapping, set to item mapping
        self.item_list = []
        self.setid2item = {}  # key:set_id+index, values: item_id
        self.category2item = {}
        self.item2category = {}
        self.outfit_info = json.load(open(annotation_path, 'r'))

        for outfit in self.outfit_info:
            items = outfit['items']
            set_id = outfit['set_id']
            for item in items:
                item_id = item['item_id']
                # get category info
                category = self.meta_data[item_id]['semantic_category']
                self.item2category[item_id] = category
                if category not in self.category2item:
                    self.category2item[category] = {}

                if set_id not in self.category2item[category]:
                    self.category2item[category][set_id] = []

                self.category2item[category][set_id].append(item_id)
                self.setid2item[set_id + '_' + str(item['index'])] = item_id
                if item_id not in self.item_list:
                    self.item_list.append(item_id)

        self.item2index = {}
        for i, item in enumerate(self.item_list):
            self.item2index[item] = i

        if typespaces_fn is not None:
            self.type_spaces = self.load_typespaces(typespaces_fn)

        self.train = train
        # collect positive pairs
        if self.train:
            self.pos_pairs = self.collect_pos_pairs()

            # read text features
            if text_feat_path is not None:
                self.text_feat_dim = text_feat_dim
                # get feature vector based on text description
                self.desc2vecs = {}
                with open(text_feat_path, 'r') as rf:
                    for line in rf:
                        line = line.strip()
                        if not line:
                            continue

                        vec = line.split(',')
                        label = ','.join(vec[:-self.text_feat_dim])
                        vec = np.array(
                            [float(x) for x in vec[-self.text_feat_dim:]],
                            np.float32)
                        assert (len(vec) == text_feat_dim)
                        self.desc2vecs[label] = vec

                # get item to text description mapping
                self.item2desc = {}
                for item in self.item_list:
                    desc = self.meta_data[item]['title']
                    if not desc:
                        desc = self.meta_data[item]['url_name']
                    desc = desc.replace('\n',
                                        ',').encode('ascii',
                                                    'ignore').strip().lower()
                    if desc and desc in self.desc2vecs:
                        self.item2desc[item] = desc

        else:  # test data setting
            if compatibility_test_fn is not None:
                self.compatibility_questions = \
                    self.collect_compatibility_questions(compatibility_test_fn)
            if fitb_test_fn is not None:
                self.fitb_questions = self.collect_fitb_questions(fitb_test_fn)

    def collect_pos_pairs(self):
        '''prepare positive pairs when training'''
        pos_pairs = []
        max_items = 0
        for outfit in self.outfit_info:
            items = outfit['items']
            set_len = len(items)
            max_items = max(set_len, max_items)
            set_id = outfit['set_id']
            for j in range(set_len - 1):
                for k in range(j + 1, set_len):
                    pos_pairs.append(
                        [set_id, items[j]['item_id'], items[k]['item_id']])
        return pos_pairs

    @staticmethod
    def load_typespaces(typespace_fn):
        """Loads a mapping of pairs of types to the embedding used to compare
            them.

        Args:
            rand_typespaces: Boolean indicator of randomly assigning type
                specific spaces to their embedding.
            num_rand_embed: number of embeddings to use when rand_typespaces
                is true.
        """
        typespaces = pickle.load(open(typespace_fn, 'rb'))

        ts = {}
        for index, t in enumerate(typespaces):
            ts[t] = index
        typespaces = ts
        return typespaces

    @staticmethod
    def parse_iminfo(question, im2index, id2im, gt=None):
        """This is the same with "Learning Type-Aware Embeddings for Fashion
            Compatibility" https://github.com/mvasil/fashion-compatibility

            Maps the questions from the FITB and compatibility tasks back to
            their index in the precomputed matrix of features

        Args:
            question: List of images to measure compatibility
            im2index: Dictionary mapping an image name to its location in a
                      precomputed matrix of features
            gt: optional, the ground truth outfit set this item belongs to
        """
        questions = []
        is_correct = np.zeros(len(question), np.bool)
        for index, im_id in enumerate(question):
            set_id = im_id.split('_')[0]
            if gt is None:
                gt = set_id

            im = id2im[im_id]
            questions.append((im2index[im], im))
            is_correct[index] = set_id == gt
        return questions, is_correct, gt

    def collect_fitb_questions(self, fitb_test_fn):
        ''' collect Fill-in-the-blank questions'''
        fitb_questions = []
        data = json.load(open(fitb_test_fn, 'r'))
        for item in data:
            question = item['question']
            q_index, _, gt = self.parse_iminfo(question, self.item2index,
                                               self.setid2item)
            answer = item['answers']
            a_index, is_correct, _ = self.parse_iminfo(answer, self.item2index,
                                                       self.setid2item, gt)
            fitb_questions.append((q_index, a_index, is_correct))

        return fitb_questions

    def collect_compatibility_questions(self, compatibility_test_fn):
        '''collect compatibility test question'''
        compatibility_rf = open(compatibility_test_fn).readlines()
        compatibility_questions = []
        for line in compatibility_rf:
            data = line.strip().split()
            compat_question, _, _ = self.parse_iminfo(data[1:],
                                                      self.item2index,
                                                      self.setid2item)
            compatibility_questions.append((compat_question, int(data[0])))
        return compatibility_questions

    def get_single_compatibility_score(self,
                                       embeds,
                                       item_ids,
                                       metric,
                                       use_cuda=True):
        n_items = embeds.size(0)
        outfit_score = 0.0
        num_comparisons = 0.0
        for i in range(n_items - 1):
            item1_id = item_ids[i]
            type1 = self.item2category[item1_id]
            for j in range(i + 1, n_items):
                item2_id = item_ids[j]
                type2 = self.item2category[item2_id]
                condition = self.get_typespaces(type1, type2)
                embed1 = embeds[i][condition].unsqueeze(0)
                embed2 = embeds[j][condition].unsqueeze(0)
                if use_cuda:
                    embed1 = embed1.cuda()
                    embed2 = embed2.cuda()
                if metric is None:
                    outfit_score += torch.nn.functional.pairwise_distance(
                        embed1, embed2, 2)
                else:
                    outfit_score += metric(Variable(embed1 * embed2)).data
                num_comparisons += 1
        outfit_score /= num_comparisons
        outfit_score = 1 - outfit_score.item()
        return outfit_score

    def test_compatibility(self, embeds, metric):
        """ Returns the area under a roc curve for the compatibility task
            embeds: precomputed embedding features used to score
                    each compatibility question
            metric: a function used to score the elementwise product
                    of a pair of embeddings, if None euclidean
                    distance is used
        """
        scores = []
        labels = np.zeros(len(self.compatibility_questions), np.int32)
        for idx, (outfit, label) in enumerate(self.compatibility_questions):
            labels[idx] = label
            n_items = len(outfit)
            outfit_score = 0.0
            num_comparisons = 0.0

            for i in range(n_items - 1):
                item1, item1_id = outfit[i]
                type1 = self.item2category[item1_id]
                for j in range(i + 1, n_items):
                    item2, item2_id = outfit[j]
                    type2 = self.item2category[item2_id]
                    condition = self.get_typespaces(type1, type2)
                    embed1 = embeds[item1][condition].unsqueeze(0)
                    embed2 = embeds[item2][condition].unsqueeze(0)
                    embed1 = embed1.cuda()
                    embed2 = embed2.cuda()
                    if metric is None:
                        outfit_score += torch.nn.functional.pairwise_distance(
                            embed1, embed2, 2)
                    else:
                        outfit_score += metric(Variable(embed1 * embed2)).data

                    num_comparisons += 1

            outfit_score /= num_comparisons
            scores.append(outfit_score)

        scores = torch.cat(scores).squeeze().cpu().numpy()
        scores = 1 - scores
        auc = roc_auc_score(labels, scores)
        return auc

    def test_fitb(self, embeds, metric):
        """Returns the accuracy of the fill in the blank task

        Args:
            embeds: precomputed embedding features used to score each
                compatibility question.
            metric: a function used to score the elementwise product of a pair
                of embeddings, if None euclidean distance is used.
        """
        correct = 0.
        n_questions = 0.
        for q_index, (questions, answers,
                      is_correct) in enumerate(self.fitb_questions):
            answer_score = np.zeros(len(answers), dtype=np.float32)

            for index, (answer, item_id1) in enumerate(answers):
                type1 = self.item2category[item_id1]
                score = 0.0

                for question, item_id2 in questions:
                    type2 = self.item2category[item_id2]
                    condition = self.get_typespaces(type1, type2)
                    embed1 = embeds[question][condition].unsqueeze(0)
                    embed2 = embeds[answer][condition].unsqueeze(0)
                    embed1 = embed1.cuda()
                    embed2 = embed2.cuda()

                    if metric is None:
                        score += torch.nn.functional.pairwise_distance(
                            embed1, embed2, 2)
                    else:
                        score += metric(Variable(embed1 * embed2)).data
                answer_score[index] = score.squeeze().cpu().numpy()

            # scores are based on distances so need to convert them so higher
            # is better
            correct += is_correct[np.argmin(answer_score)]
            n_questions += 1

        acc = correct / n_questions
        return acc

    def load_train_item(self, item_id):
        ''' Returns a single item in the triplet and its data'''
        img_path = os.path.join(self.img_path, item_id + '.jpg')
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        if item_id in self.item2desc:
            text = self.item2desc[item_id]
            text_features = self.desc2vecs[text]
            has_text = 1
        else:
            text_features = np.zeros(self.text_feat_dim, np.float32)
            has_text = 0

        has_text = np.float32(has_text)
        item_cate = self.item2category[item_id]
        return img, text_features, has_text, item_cate

    def sample_negative(self, item_id, item_cate):
        '''Returns a randomly sampled item from a different set
            than the outfit at data_index, but of the same type as item_type
        '''
        item_out = item_id
        candidate_sets = list(self.category2item[item_cate].keys())
        attempts = 0
        while item_out == item_id and attempts < 100:
            choice = np.random.choice(candidate_sets)
            items = self.category2item[item_cate][choice]
            item_index = np.random.choice(range(len(items)))
            item_out = items[item_index]
            attempts += 1
        return item_out

    def get_typespaces(self, anchor, pair):
        ''' Returns the index of the category specific embedding
            for the pair of item types provided as input
        '''
        query = (anchor, pair)
        if query not in self.type_spaces:
            query = (pair, anchor)
        return self.type_spaces[query]

    def __getitem__(self, index):
        if self.train:
            set_id, anchor_im, pos_im = self.pos_pairs[index]
            img1, desc1, has_text1, anchor_cate = self.load_train_item(
                anchor_im)
            img2, desc2, has_text2, pos_cate = self.load_train_item(pos_im)

            neg_im = self.sample_negative(pos_im, pos_cate)
            img3, desc3, has_text3, _ = self.load_train_item(neg_im)
            condition = self.get_typespaces(anchor_cate, pos_cate)

            return {
                'img': img1,
                'text': desc1,
                'has_text': has_text1,
                'pos_img': img2,
                'pos_text': desc2,
                'pos_has_text': has_text2,
                'neg_img': img3,
                'neg_text': desc3,
                'neg_has_text': has_text3,
                'condition': condition
            }
        else:
            item_id = self.item_list[index]
            img_path = os.path.join(self.img_path, item_id + '.jpg')
            img = Image.open(img_path).convert('RGB')
            img = self.transform(img)
            return {'img': img}

    def shuffle(self):
        np.random.shuffle(self.pos_pairs)

    def __len__(self):
        if self.train:
            return len(self.pos_pairs)
        else:
            return len(self.item_list)
