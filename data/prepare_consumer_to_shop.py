import os

PREFIX = 'Consumer_to_shop/Anno'


def split_img():
    fn = open('Consumer_to_shop/Eval/list_eval_partition.txt').readlines()

    # train dataset
    train_consumer2shop = open(
        os.path.join(PREFIX, 'train_consumer2shop.txt'), 'w')
    train_imgs = []

    # test dataset
    test_consumer = open(os.path.join(PREFIX, 'consumer.txt'), 'w')
    test_shop = open(os.path.join(PREFIX, 'shop.txt'), 'w')
    consumer_imgs, shop_imgs = [], []

    for i, line in enumerate(fn[2:]):
        aline = line.strip('\n').split()
        consumer, shop, _, cate = aline[0], aline[1], aline[2], aline[3]
        if cate == 'train':
            newline = consumer + ' ' + shop + '\n'
            train_consumer2shop.write(newline)
            train_imgs.append(consumer)
            train_imgs.append(shop)
        else:
            newline = consumer + '\n'
            test_consumer.write(newline)
            newline = shop + '\n'
            test_shop.write(newline)
            consumer_imgs.append(consumer)
            shop_imgs.append(shop)

    train_consumer2shop.close()
    test_consumer.close()
    test_shop.close()
    return train_imgs, consumer_imgs, shop_imgs


def split_bbox(train_set, consumer_set, shop_set):
    rf = open(os.path.join(PREFIX, 'list_bbox_consumer2shop.txt')).readlines()
    img2bbox = {}
    for i, line in enumerate(rf[2:]):
        aline = line.strip('\n').split()
        img = aline[0]
        bbox = aline[-4:]
        img2bbox[img] = bbox

    wf_train = open(os.path.join(PREFIX, 'list_bbox_train.txt'), 'w')
    wf_consumer = open(os.path.join(PREFIX, 'list_bbox_consumer.txt'), 'w')
    wf_shop = open(os.path.join(PREFIX, 'list_bbox_shop.txt'), 'w')
    for i, img in enumerate(train_set):
        bbox = img2bbox[img]
        newline = img + ' ' + bbox[0] + ' ' + bbox[1] + ' ' + bbox[
            2] + ' ' + bbox[3] + '\n'
        wf_train.write(newline)

    for i, img in enumerate(consumer_set):
        bbox = img2bbox[img]
        newline = img + ' ' + bbox[0] + ' ' + bbox[1] + ' ' + bbox[
            2] + ' ' + bbox[3] + '\n'
        wf_consumer.write(newline)

    for i, img in enumerate(shop_set):
        bbox = img2bbox[img]
        newline = img + ' ' + bbox[0] + ' ' + bbox[1] + ' ' + bbox[
            2] + ' ' + bbox[3] + '\n'
        wf_shop.write(newline)

    wf_train.close()
    wf_consumer.close()
    wf_shop.close()


def split_ids(train_set, consumer_set, shop_set):
    id2label = dict()
    rf = open(os.path.join(PREFIX, 'list_item_consumer2shop.txt')).readlines()
    for i, line in enumerate(rf[1:]):
        id2label[line.strip('\n')] = i

    def write_id(cloth, wf):
        for i, line in enumerate(cloth):
            id = line.strip('\n').split('/')[3]
            label = id2label[id]
            wf.write('%s\n' % str(label))
        wf.close()

    train_id = open(os.path.join(PREFIX, 'train_id.txt'), 'w')
    consumer_id = open(os.path.join(PREFIX, 'consumer_id.txt'), 'w')
    shop_id = open(os.path.join(PREFIX, 'shop_id.txt'), 'w')
    write_id(train_set, train_id)
    write_id(consumer_set, consumer_id)
    write_id(shop_set, shop_id)


def split_lms(train_set, consumer_set, shop_set):
    rf = open(os.path.join(PREFIX,
                           'list_landmarks_consumer2shop.txt')).readlines()
    img2landmarks = {}
    for i, line in enumerate(rf[2:]):
        aline = line.strip('\n').split()
        img = aline[0]
        landmarks = aline[3:]
        img2landmarks[img] = landmarks

    wf_train = open(os.path.join(PREFIX, 'list_landmarks_train.txt'), 'w')
    wf_consumer = open(
        os.path.join(PREFIX, 'list_landmarks_consumer.txt'), 'w')
    wf_shop = open(os.path.join(PREFIX, 'list_landmarks_shop.txt'), 'w')

    def write_landmarks(img_set, wf):
        for i, img in enumerate(img_set):
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

            while len(one_lms) < 16:  # 8 pairs
                one_lms.append('000')

            wf.write(img)
            wf.write(' ')
            for lm in one_lms:
                wf.write(lm)
                wf.write(' ')
            wf.write('\n')
        wf.close()

    write_landmarks(train_set, wf_train)
    write_landmarks(consumer_set, wf_consumer)
    write_landmarks(shop_set, wf_shop)


if __name__ == '__main__':
    train_imgs, consumer_imgs, shop_imgs = split_img()
    split_bbox(train_imgs, consumer_imgs, shop_imgs)
    split_ids(train_imgs, consumer_imgs, shop_imgs)
    split_lms(train_imgs, consumer_imgs, shop_imgs)
