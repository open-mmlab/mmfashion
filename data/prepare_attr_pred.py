import os

PREFIX = 'Attr_Predict/Anno'


def split_img():
    rf = open('Attr_Predict/Eval/list_eval_partition.txt').readlines()
    wf1 = open(os.path.join(PREFIX, 'train.txt'), 'w')
    wf2 = open(os.path.join(PREFIX, 'test.txt'), 'w')
    wf3 = open(os.path.join(PREFIX, 'val.txt'), 'w')

    for i, line in enumerate(rf[2:]):
        aline = line.strip('\n').split()
        imgname, prefix = aline[0], aline[1]
        if prefix == 'train':
            wf1.write('%s\n' % imgname)
        elif prefix == 'test':
            wf2.write('%s\n' % imgname)
        else:
            wf3.write('%s\n' % imgname)

    wf1.close()
    wf2.close()
    wf3.close()


def split_attribute(train_img, test_img, val_img):
    rf = open(os.path.join(PREFIX, 'list_attr_img.txt')).readlines()
    wf1 = open(os.path.join(PREFIX, 'train_attr.txt'), 'w')
    wf2 = open(os.path.join(PREFIX, 'test_attr.txt'), 'w')
    wf3 = open(os.path.join(PREFIX, 'val_attr.txt'), 'w')

    def sort_attr():
        d = dict()
        for i, line in enumerate(rf[2:]):
            aline = line.strip('\n').split()
            imgname = aline[0]
            d[imgname] = []
            attrs = aline[1:]
            for ai, a in enumerate(attrs):
                if int(a) <= 0:
                    d[imgname].append(0)
                else:
                    d[imgname].append(int(a))
        return d

    attributes = sort_attr()

    def write_attr(imgs, wf):
        for i, line in enumerate(imgs):
            imgname = line.strip('\n')
            attr = attributes[imgname]
            for a in attr:
                wf.write('%s ' % str(a))
            wf.write('\n')
        wf.close()

    write_attr(train_img, wf1)
    write_attr(test_img, wf2)
    write_attr(val_img, wf3)


def split_bbox(train_img, test_img, val_img):
    rf = open(os.path.join(PREFIX, 'list_bbox.txt')).readlines()
    wf1 = open(os.path.join(PREFIX, 'train_bbox.txt'), 'w')
    wf2 = open(os.path.join(PREFIX, 'test_bbox.txt'), 'w')
    wf3 = open(os.path.join(PREFIX, 'val_bbox.txt'), 'w')

    def sort_bbox():
        d = dict()
        for i, line in enumerate(rf[2:]):
            aline = line.strip('\n').split()
            imgname = aline[0]
            d[imgname] = []
            bbox = aline[1:]
            for bi, b in enumerate(bbox):
                d[imgname].append(int(b))
        return d

    bboxes = sort_bbox()

    def write_bbox(imgs, wf):
        for i, line in enumerate(imgs):
            imgname = line.strip('\n')
            bbox = bboxes[imgname]
            for b in bbox:
                wf.write('%s ' % str(b))
            wf.write('\n')
        wf.close()

    write_bbox(train_img, wf1)
    write_bbox(test_img, wf2)
    write_bbox(val_img, wf3)


def split_category(train_img, test_img, val_img):
    train_img = open(os.path.join(PREFIX, 'train.txt')).readlines()
    test_img = open(os.path.join(PREFIX, 'test.txt')).readlines()
    val_img = open(os.path.join(PREFIX, 'val.txt')).readlines()

    wf1 = open(os.path.join(PREFIX, 'train_cate.txt'), 'w')
    wf2 = open(os.path.join(PREFIX, 'test_cate.txt'), 'w')
    wf3 = open(os.path.join(PREFIX, 'val_cate.txt'), 'w')

    def gather_cate():
        d = dict()
        rf = open(os.path.join(PREFIX, 'list_category_img.txt')).readlines()
        for i, line in enumerate(rf[2:]):
            aline = line.strip('\n').split()
            imgname, cate = aline[0], aline[1]
            d[imgname] = cate

        return d

    img2cate = gather_cate()

    def write_cate(imgs, wf):
        # cate_ids = []
        for i, line in enumerate(imgs):
            imgname = line.strip('\n').split()[0]
            cate_id = int(img2cate[imgname])
            # cate_ids.append(cate_id)
            wf.write('%d\n' % cate_id)

    write_cate(train_img, wf1)
    write_cate(test_img, wf2)
    write_cate(val_img, wf3)


def split_lms(train_img, test_img, val_img):
    rf = open(os.path.join(PREFIX, 'list_landmarks.txt')).readlines()
    wf1 = open(os.path.join(PREFIX, 'train_landmarks.txt'), 'w')
    wf2 = open(os.path.join(PREFIX, 'test_landmarks.txt'), 'w')
    wf3 = open(os.path.join(PREFIX, 'val_landmarks.txt'), 'w')

    def sort_lm():
        d = dict()
        for i, line in enumerate(rf[2:]):
            aline = line.strip('\n').split()
            imgname = aline[0]
            d[imgname] = []
            lms = aline[2:]
            for li, lm in enumerate(lms):
                if li % 3 == 0:
                    continue
                else:
                    d[imgname].append(int(lm))
        return d

    landmarks = sort_lm()

    def write_lm(imgs, wf):
        for i, line in enumerate(imgs):
            imgname = line.strip('\n')
            lm = landmarks[imgname]
            for a in lm:
                wf.write('%s ' % str(a))

            num_lm = len(lm)
            if num_lm < 16:
                for cnt in range(16 - num_lm):
                    wf.write('0 ')
            wf.write('\n')
        wf.close()

    write_lm(train_img, wf1)
    write_lm(test_img, wf2)
    write_lm(val_img, wf3)


if __name__ == '__main__':
    split_img()
    train_img = open(os.path.join(PREFIX, 'train.txt')).readlines()
    test_img = open(os.path.join(PREFIX, 'test.txt')).readlines()
    val_img = open(os.path.join(PREFIX, 'val.txt')).readlines()

    split_attribute(train_img, test_img, val_img)
    split_category(train_img, test_img, val_img)
    split_bbox(train_img, test_img, val_img)
    split_lms(train_img, test_img, val_img)
