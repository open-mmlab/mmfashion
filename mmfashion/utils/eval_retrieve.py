import scipy.io as sio
from scipy.spatial.distance import cdist as cdist
import numpy as np

def _calculate(idxes, idx2id, query_id):
    ids = []
    cnt = 0
    for i in idxes:
        ids.append(idx2id[i])

    if query_id in ids:
       cnt += 1
    return cnt


def show_result(query_embeds, gallery_embeds, query_dict, gallery_dict):
    top3, top5, top10, top20 = 0, 0, 0,0
    total = 0

    for qi, query_embed in enumerate(query_embeds):
        dist = []
        for gi, gallery_embed in enumerate(gallery_embeds):
            one_dist = cdist(query_embed.reshape(1,-1), gallery_embed.reshape(1,-1), 'euclidean')
            dist.append(one_dist[0][0])
        dist = np.asarray(dist)
        order = np.argsort(dist)
        query_id = query_dict[qi]
        
        print('query_id', query_id)

        top3 += _calculate(order[:3], gallery_dict, query_id)
        top5 += _calculate(order[:5], gallery_dict, query_id)
        top10 += _calculate(order[:10], gallery_dict, query_id)
        top20 += _calculate(order[:20], gallery_dict, query_id)
        total += 1
        if qi % 10==0:
           print(top3, top5, top10, top20)
           acc3, acc5, acc10, acc20 = 100*float(top3)/ total, 100*float(top5)/ total, 100*float(top10)/ total, 100*float(top20)/ total
           print('top3 = %.4f, top5 = %.4f, top10 = %.4f, top20 = %.4f '%
                (acc3, acc5, acc10, acc20))

    print('------------- Recall Rate ------------------')
    print(top3, top5, top10, top20)
    acc3, acc5, acc10, acc20 = 100*float(top3)/ total, 100*float(top5)/ total, 100*float(top10)/ total, 100*float(top20)/ total
    print('top3 = %.4f, top5 = %.4f, top10 = %.4f, top20 = %.4f '%
            (acc3, acc5, acc10, acc20))


def load_dict(toread):
    dic= dict()
    rf = open(toread).readlines()
    for i, line in enumerate(rf):
        aline = line.strip('\n')
        dic[i] = int(aline)
    return dic

query_dict = load_dict('/newDisk/users/liuxin/mmfashion-prerelease/mmfashion/datasets/In-shop/Anno/query_idx2id.txt')
gallery_dict = load_dict('/newDisk/users/liuxin/mmfashion-prerelease/mmfashion/datasets/In-shop/Anno/gallery_idx2id.txt')

query_embeds = sio.loadmat('query_embeds.mat')['embeds']
gallery_embeds = sio.loadmat('gallery_embeds.mat')['embeds']

print('query_embeds shape', query_embeds.shape)
print('gallery_embeds shape', gallery_embeds.shape)


show_result(query_embeds, gallery_embeds, query_dict, gallery_dict)
