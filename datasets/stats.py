import os
from nltk.corpus import wordnet as wn

from datasets.pascalvoc import VOCDetection
from datasets.mscoco import COCODetection
from datasets.imgnetdet import ImageNetDetection
from datasets.imgnetvid import ImageNetVidDetection


splits = ['val']

for split in splits:
    print(split)
    dsets = list()
    if split == 'train':
        dsets.append(('voc trainval 07+12',
                      VOCDetection(root=os.path.join('datasets', 'PascalVOC', 'VOCdevkit'),
                                   splits=[(2007, 'trainval'), (2012, 'trainval')])))

        dsets.append(('coco train 17',
                      COCODetection(root=os.path.join('datasets', 'MSCoco'),
                                    splits=['instances_train2017'], use_crowd=False)))

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
                                    splits=['instances_val2017'])))

        dsets.append(('det val',
                      ImageNetDetection(root=os.path.join('datasets', 'ImageNetDET', 'ILSVRC'),
                                        splits=['val'], allow_empty=False)))

        dsets.append(('vid val',
                      ImageNetVidDetection(root=os.path.join('datasets', 'ImageNetVID', 'ILSVRC'),
                                           splits=[(2017, 'val')], allow_empty=False)))

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

    # used to make table on github
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

    # used to make table in thesis
    yet_to_do = classes_order
    new_order = list()
    in_set = dict()
    while len(yet_to_do) > 0:
        print(len(yet_to_do))
        for cls in yet_to_do:
            sets = list(classes[cls].keys())
            if 'vid val' in sets:
                if 'voc test 07' in sets:
                    if 'coco val 17' in sets:
                        if 'det val' in sets:
                            new_order.append(cls)
                            in_set[cls] = [1,1,1,1]
                            yet_to_do.remove(cls)
                            break
                        else:
                            new_order.append(cls)
                            in_set[cls] = [1,1,1,0]
                            yet_to_do.remove(cls)
                            break
                    else:
                        if 'det val' in sets:
                            new_order.append(cls)
                            in_set[cls] = [1,1,0,1]
                            yet_to_do.remove(cls)
                            break
                        else:
                            new_order.append(cls)
                            in_set[cls] = [1,1,0,0]
                            yet_to_do.remove(cls)
                            break
                else:
                    if 'coco val 17' in sets:
                        if 'det val' in sets:
                            new_order.append(cls)
                            in_set[cls] = [1,0,1,1]
                            yet_to_do.remove(cls)
                            break
                        else:
                            new_order.append(cls)
                            in_set[cls] = [1,0,1,0]
                            yet_to_do.remove(cls)
                            break
                    else:
                        if 'det val' in sets:
                            new_order.append(cls)
                            in_set[cls] = [1,0,0,1]
                            yet_to_do.remove(cls)
                            break
                        else:
                            new_order.append(cls)
                            in_set[cls] = [1,0,0,0]
                            yet_to_do.remove(cls)
                            break
            else:
                if 'voc test 07' in sets:
                    if 'coco val 17' in sets:
                        if 'det val' in sets:
                            new_order.append(cls)
                            in_set[cls] = [0,1,1,1]
                            yet_to_do.remove(cls)
                            break
                        else:
                            new_order.append(cls)
                            in_set[cls] = [0,1,1,0]
                            yet_to_do.remove(cls)
                            break
                    else:
                        if 'det val' in sets:
                            new_order.append(cls)
                            in_set[cls] = [0,1,0,1]
                            yet_to_do.remove(cls)
                            break
                        else:
                            new_order.append(cls)
                            in_set[cls] = [0,1,0,0]
                            yet_to_do.remove(cls)
                            break
                else:
                    if 'coco val 17' in sets:
                        if 'det val' in sets:
                            new_order.append(cls)
                            in_set[cls] = [0,0,1,1]
                            yet_to_do.remove(cls)
                            break
                        else:
                            new_order.append(cls)
                            in_set[cls] = [0,0,1,0]
                            yet_to_do.remove(cls)
                            break
                    else:
                        if 'det val' in sets:
                            new_order.append(cls)
                            in_set[cls] = [0,0,0,1]
                            yet_to_do.remove(cls)
                            break
                        else:
                            new_order.append(cls)
                            in_set[cls] = [0,0,0,0]
                            yet_to_do.remove(cls)
                            break

    # for cls in new_order:
    #     str = wn.synset_from_pos_and_offset('n', int(cls[1:]))._name.split('.n.')[0]
    #     if in_set[cls][0] == 0 and in_set[cls][1] == 0 and in_set[cls][2] == 0 and in_set[cls][3] == 1:
    #         for i in in_set[cls][3:]:
    #             if i:
    #                 str += ' & \ding{51}'
    #             else:
    #                 str += ' &'
    #         print(str + ' \\\\')

    last = list()
    for cls in new_order:
        if in_set[cls][0] == 0 and in_set[cls][1] == 0 and in_set[cls][2] == 0 and in_set[cls][3] == 1:
            last.append(cls)

    num = 50
    for ci in range(num):
        str = ''
        for add in range(5):
            if ci+add*num < len(last):
                cls = last[ci+add*num]
                str += wn.synset_from_pos_and_offset('n', int(cls[1:]))._name.split('.n.')[0].replace('_', '\\_') + ' & '
        print(str)
    print()
