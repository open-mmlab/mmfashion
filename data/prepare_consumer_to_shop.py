import os

PREFIX = 'Consumer_to_shop/Anno'

def split_img():
    fn = open('Consumer_to_shop/Eval/list_eval_partition.txt').readlines()

    # train dataset
    train_consumer2shop = open(os.path.join(PREFIX, 'train_consumer2shop.txt'), 'w')
    train_imgs = []

    # test dataset
    test_consumer2shop = open(os.path.join(PREFIX, 'test_consumer2shop.txt'), 'w')
    test_imgs = []

    for i, line in enumerate(fn[2:]):
        aline = line.strip('\n').split()
        consumer, shop, id, cate = aline[0], aline[1], aline[2], aline[3]
        newline = consumer+' '+shop+'\n'
        if cate == 'train':
            train_consumer2shop.write(newline)
            train_imgs.append(consumer)
            train_imgs.append(shop)
        else:
            test_consumer2shop.write(newline)
            test_imgs.append(consumer)
            test_imgs.append(shop)

    train_consumer2shop.close()
    test_consumer2shop.close()
    return train_imgs, test_imgs


def split_bbox(train_set, test_set):
    rf = open(os.path.join(PREFIX, 'list_bbox_consumer2shop.txt')).readlines()
    img2bbox = {}
    for i, line in enumerate(rf[2:]):
        aline = line.strip('\n').split()
        img = aline[0]
        bbox = aline[-4:]
        img2bbox[img] = bbox

    wf_train = open(os.path.join(PREFIX, 'list_bbox_train.txt'), 'w')
    wf_test = open(os.path.join(PREFIX, 'list_bbox_test.txt'), 'w')
    
    for i, img in enumerate(train_set):
        bbox = img2bbox[img]
        newline = img+' '+bbox[0]+' '+bbox[1]+' '+bbox[2]+' '+bbox[3]+'\n'
        wf_train.write(newline)

    for i, img in enumerate(test_set):
        bbox = img2bbox[img]
        newline = img+' '+bbox[0]+' '+bbox[1]+' '+bbox[2]+' '+bbox[3]+'\n'
        wf_test.write(newline)

    wf_train.close()
    wf_test.close()

def split_lms(train_set, test_set):
    rf = open(os.path.join(PREFIX, 'list_landmarks_consumer2shop.txt')).readlines()
    img2landmarks = {}

    for i, line in enumerate(rf[2:]):
        aline = line.strip('\n').split()
        img = aline[0]
        landmarks = aline[3:]
        img2landmarks[img] = landmarks

    wf_train = open(os.path.join(PREFIX, 'list_landmarks_train.txt'), 'w')
    wf_test = open(os.path.join(PREFIX, 'list_landmarks_test.txt'), 'w')

    for i, img in enumerate(train_set):
        landmarks = img2landmarks[img]
        one_lms = []
        for j, lm in enumerate(landmarks):
            if j%3==0 : # visibility
                if lm==0: # visible
                    one_lms.append(landmarks[j+1])
                    one_lms.append(landmarks[j+2])
                else: # invisible or truncated
                    one_lms.append('000')
                    one_lms.append('000')

        while len(one_lms)<16:
            one_lms.append('000')

        wf_train.write(img)
        wf_train.write(' ')
        for lm in one_lms:
            wf_train.write(lm)
            wf_train.write(' ')
        wf_train.write('\n')
    wf_train.close()

    for i, img in enumerate(test_set):
        landmarks = img2landmarks[img]
        one_lms = []
        for j, lm in enumerate(landmarks):
            if j % 3 == 0:  # visibility
                if lm == '0':  # visible
                    one_lms.append(landmarks[j + 1])
                    one_lms.append(landmarks[j + 2])
                else:  # invisible or truncated
                    one_lms.append('000')
                    one_lms.append('000')

        while len(one_lms) < 16:
            one_lms.append('000')

        wf_test.write(img)
        wf_test.write(' ')
        for lm in one_lms:
            wf_test.write(lm)
            wf_test.write(' ')
        wf_test.write('\n')
    wf_test.close()


if __name__ == '__main__':
    train_imgs, test_imgs = split_img()
    split_bbox(train_imgs, test_imgs)
    split_lms(train_imgs, test_imgs)