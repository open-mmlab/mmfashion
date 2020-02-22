import os

PREFIX = 'In-shop/Anno'


def split_img():
    fn = open('In-shop/Eval/list_eval_partition.txt').readlines()
    train = open(os.path.join(PREFIX, 'train_img.txt'), 'w')
    query = open(os.path.join(PREFIX, 'query_img.txt'), 'w')
    gallery = open(os.path.join(PREFIX, 'gallery_img.txt'), 'w')

    for i, line in enumerate(fn[2:]):
        aline = line.strip('\n').split()
        img, _, prefix = aline[0], aline[1], aline[2]
        if prefix == 'train':
            train.write(img)
            train.write('\n')
        else:
            if prefix == 'query':
                query.write(img)
                query.write('\n')

            elif prefix == 'gallery':
                gallery.write(img)
                gallery.write('\n')

    train.close()
    query.close()
    gallery.close()


def split_label():
    id2label = {}
    labelf = open(os.path.join(PREFIX, 'list_attr_items.txt')).readlines()
    for line in labelf[2:]:
        aline = line.strip('\n').split()
        id, label = aline[0], aline[1:]
        id2label[id] = label

    def get_label(fn, prefix):
        rf = open(fn).readlines()
        wf = open(os.path.join(PREFIX, '%s_labels.txt' % prefix), 'w')
        for line in rf:
            aline = line.strip('\n').split('/')
            id = aline[3]
            label = id2label[id]
            for element in label:
                if element == '1':
                    wf.write('1 ')
                else:
                    wf.write('0 ')
            wf.write('\n')
        wf.close()

    get_label(os.path.join(PREFIX, 'train_img.txt'), 'train')
    get_label(os.path.join(PREFIX, 'gallery_img.txt'), 'gallery')
    get_label(os.path.join(PREFIX, 'query_img.txt'), 'query')


def split_ids():
    id2label = dict()
    rf = open(os.path.join(PREFIX, 'list_item_inshop.txt')).readlines()
    for i, line in enumerate(rf[1:]):
        id2label[line.strip('\n')] = i

    def write_id(rf, wf):
        for i, line in enumerate(rf):
            id = line.strip('\n').split('/')[3]
            label = id2label[id]
            wf.write('%s\n' % str(label))
        wf.close()

    rf1 = open(os.path.join(PREFIX, 'train_img.txt')).readlines()
    rf2 = open(os.path.join(PREFIX, 'query_img.txt')).readlines()
    rf3 = open(os.path.join(PREFIX, 'gallery_img.txt')).readlines()
    wf1 = open(os.path.join(PREFIX, 'train_id.txt'), 'w')
    wf2 = open(os.path.join(PREFIX, 'query_id.txt'), 'w')
    wf3 = open(os.path.join(PREFIX, 'gallery_id.txt'), 'w')
    write_id(rf1, wf1)
    write_id(rf2, wf2)
    write_id(rf3, wf3)


def split_bbox():
    name2bbox = {}
    rf = open(os.path.join(PREFIX, 'list_bbox_inshop.txt')).readlines()

    for line in rf[2:]:
        aline = line.strip('\n').split()
        name = aline[0]
        bbox = [aline[3], aline[4], aline[5], aline[6]]
        name2bbox[name] = bbox

    def get_bbox(fn, prefix):
        namef = open(fn).readlines()
        wf = open(os.path.join(PREFIX, '%s_bbox.txt' % prefix), 'w')
        for i, name in enumerate(namef):
            name = name.strip('\n')
            bbox = name2bbox[name]
            for cor in bbox:
                wf.write('%s ' % cor)
            wf.write('\n')
        wf.close()

    get_bbox(os.path.join(PREFIX, 'train_img.txt'), 'train')
    get_bbox(os.path.join(PREFIX, 'gallery_img.txt'), 'gallery')
    get_bbox(os.path.join(PREFIX, 'query_img.txt'), 'query')


def split_lms():
    name2lm = {}
    rf = open(os.path.join(PREFIX, 'list_landmarks_inshop.txt')).readlines()
    for line in rf[2:]:
        aline = line.strip('\n').split()
        name = aline[0]
        landmark = []
        for j, element in enumerate(aline[3:]):
            if j % 3 == 0:
                if element != '0':
                    continue
                else:
                    landmark += [aline[j + 1], aline[j + 2]]
        name2lm[name] = landmark

    def get_lm(fn, prefix):
        lines = open(fn).readlines()
        wf = open(os.path.join(PREFIX, '%s_landmarks.txt' % prefix), 'w')

        for line in lines:
            name = line.strip('\n')
            lms = name2lm[name]
            cnt = len(lms)
            while cnt < 16:
                lms.append('0')
                cnt = len(lms)
            for lm in lms:
                wf.write('%s ' % lm)
            wf.write('\n')
        wf.close()

    get_lm(os.path.join(PREFIX, 'train_img.txt'), 'train')
    get_lm(os.path.join(PREFIX, 'query_img.txt'), 'query')
    get_lm(os.path.join(PREFIX, 'gallery_img.txt'), 'gallery')


if __name__ == '__main__':
    split_img()
    split_label()
    split_bbox()
    split_lms()
    split_ids()
