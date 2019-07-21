import scipy.io as sio
from scipy.spatial import distance
import numpy as np


rf = open('/newDisk/users/liuxin/mmfashion-prerelease/mmfashion/datasets/In-shop/Eval/list_eval_partition.txt').readlines()
img2id = {}
for i, line in enumerate(rf[2:]):
    aline = line.strip('\n').split()
    imgname, idname, attr = aline[0], aline[1], aline[2]
    img2id[imgname] = idname

query_f = open('/newDisk/users/liuxin/mmfashion-prerelease/mmfashion/datasets/In-shop/Anno/query_img.txt').readlines()

def get_dict(toread):
    idx2img = {}
    for i,line in enumerate(toread):
        aline = line.strip('\n')
        idx2img[i] = aline
    return idx2img

def tocheck(idxes, tofind, tocheck_id):
    cnt = 0
    for idx in idxes:
        imgname = tofind[idx]
        img_id = img2id[imgname]
        if img_id == tocheck_id:
           cnt +=1
    return cnt
        
query_dict = get_dict(query_f)

gallery_f = open('/newDisk/users/liuxin/mmfashion-prerelease/mmfashion/datasets/In-shop/Anno/gallery_img.txt').readlines()

gallery_dict = get_dict(gallery_f)


query = sio.loadmat('query_embeds.mat')['embeds']
gallery = sio.loadmat('gallery_embeds.mat')['embeds']

print('query', query.shape)
print('gallery', gallery.shape)

total = 0
for i, embed_q in enumerate(query):
    query_img = query_dict[i]
    query_id = img2id[query_img]
    #print('query_id', query_id)
    
    total += 1
    dists = []
    for j, embed_g in enumerate(gallery):
        dist = distance.euclidean(embed_q[0], embed_g[0])
        dists.append(dist)
    dists_np = np.array(dists)
    
    top1, top3, top5, top10 = 0,0,0,0
    sorted_idxes = np.argsort(dists_np)

    top1+= tocheck(sorted_idxes[:1], gallery_dict, query_id)
    top3 += tocheck(sorted_idxes[:3], gallery_dict, query_id)
    top5 += tocheck(sorted_idxes[:5], gallery_dict, query_id)
    top10 += tocheck(sorted_idxes[:10], gallery_dict, query_id)

    if i%100==0:
       print('top1= %d, top3= %d, top5= %d, top10 = %d' %(top1, top3, top5, top10))

       print('top1= %.4f, top3= %.4f, top5= %.4f, top10= %.4f' %(float(top1)/total, float(top3)/total, float(top5)/total, float(top10)/total))

print('top1= %d, top3= %d, top5= %d, top10 = %d' %(top1, top3, top5, top10))

print('top1= %.4f, top3= %.4f, top5= %.4f, top10= %.4f' %(float(top1)/total, float(top3)/total, float(top5)/total, float(top10)/total))
