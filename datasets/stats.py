import os
from nltk.corpus import wordnet as wn

from datasets.pascalvoc import VOCDetection
from datasets.mscoco import COCODetection
from datasets.imgnetdet import ImageNetDetection
from datasets.imgnetvid import ImageNetVidDetection


splits = ['train', 'val']

for split in splits:
    print(split)
    dsets = list()
    if split == 'train':
        dsets.append(('voc trainval 07+12',
                      VOCDetection(root=os.path.join('datasets', 'PascalVOC', 'VOCdevkit'),
                                   splits=[(2007, 'trainval'), (2012, 'trainval')])))

        dsets.append(('coco train 17',
                      COCODetection(root=os.path.join('datasets', 'MSCoco'),
                                    splits='instances_train2017', use_crowd=False)))

        dsets.append(('det train',
                      ImageNetDetection(root=os.path.join('datasets', 'ImageNetDET', 'ILSVRC'),
                                        splits=['train'], allow_empty=False)))
        dsets.append(('vid train',
                      ImageNetVidDetection(root=os.path.join('datasets', 'ImageNetVID', 'ILSVRC'),
                                           splits=['train'], allow_empty=False, frames=True)))

    elif split == 'val':
        dsets.append(('voc test 07',
                      VOCDetection(root=os.path.join('datasets', 'PascalVOC', 'VOCdevkit'),
                                   splits=[(2007, 'test')])))

        dsets.append(('coco val 17',
                      COCODetection(root=os.path.join('datasets', 'MSCoco'),
                                    splits='instances_val2017', skip_empty=False)))

        dsets.append(('det val',
                      ImageNetDetection(root=os.path.join('datasets', 'ImageNetDET', 'ILSVRC'),
                                        splits=['val'], allow_empty=False)))

        dsets.append(('vid val',
                      ImageNetVidDetection(root=os.path.join('datasets', 'ImageNetVID', 'ILSVRC'),
                                           splits=['val'], allow_empty=False, frames=True)))

    classes = {}
    classes_order = []
    for dset in dsets:
        print('Loading {}'.format(dset[0]))
        _, cls_stats = dset[1].stats()
        for cls in cls_stats:
            if cls[1] in classes.keys():
                classes[cls[1]][dset[0]] = cls[3]
            else:
                classes[cls[1]] = {dset[0]: cls[3]}
                classes_order.append(cls[1])

    str = ''
    for cls in classes_order:
        name = wn.synset_from_pos_and_offset('n', int(cls[1:]))._name
        str += '| `{0}` | {1: <25} '.format(cls, name)
        for dset in dsets:
            if dset[0] in classes[cls]:
                str += '| {0: <8}'.format(classes[cls][dset[0]])
            else:
                str += '| {0: <8}'.format('')
        str += '|\n'

    print(str)
