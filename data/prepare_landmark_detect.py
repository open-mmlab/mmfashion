import os

PREFIX = 'Landmark_Detect/Anno'


def split_img():
    rf = open('Landmark_Detect/Eval//list_eval_partition.txt').readlines()
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


def split_landmark(train_img, test_img, val_img):
    rf = open(os.path.join(PREFIX, 'list_landmarks.txt')).readlines()
    wf1 = open(os.path.join(PREFIX, 'train_landmarks.txt'), 'w')
    wf2 = open(os.path.join(PREFIX, 'test_landmarks.txt'), 'w')
    wf3 = open(os.path.join(PREFIX, 'val_landmarks.txt'), 'w')

    def sort_landmark():
        d = dict()
        for i, line in enumerate(rf[2:]):
            aline = line.strip('\n').split()
            imgname = aline[0]
            landmarks = aline[3:]
            new_lm = []  # create new landmark container
            for lm_i, lm in enumerate(landmarks):
                if lm_i % 3 == 0:
                    if int(lm) == 0:  # visible
                        new_lm.append(1)
                    else:  # invisible or truncated off
                        new_lm.append(0)
                else:
                    new_lm.append(int(lm))

            if len(landmarks) < 24:
                for k in range(24 - len(landmarks)):
                    new_lm.append(0)

            d[imgname] = new_lm
        return d

    landmarks = sort_landmark()

    def write_landmark(imgs, wf):
        for i, line in enumerate(imgs):
            imgname = line.strip('\n')
            landmark = landmarks[imgname]
            for lm in landmark:
                wf.write('%d' % int(lm))
                wf.write(' ')
            wf.write('\n')
        wf.close()

    write_landmark(train_img, wf1)
    write_landmark(test_img, wf2)
    write_landmark(val_img, wf3)


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


if __name__ == '__main__':
    split_img()
    train_img = open(os.path.join(PREFIX, 'train.txt')).readlines()
    test_img = open(os.path.join(PREFIX, 'test.txt')).readlines()
    val_img = open(os.path.join(PREFIX, 'val.txt')).readlines()

    split_landmark(train_img, test_img, val_img)
    split_bbox(train_img, test_img, val_img)
