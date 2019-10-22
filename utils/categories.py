"""
Code for building our WordTree
"""
import os
from nltk.corpus import wordnet as wn

with open(os.path.join('datasets', 'names', '9k.tree')) as f:
    lines = f.readlines()
lines = [l.rstrip().split() for l in lines]

cats = dict()
wnd = dict()
inds = dict()
for i in range(len(lines)):
    cats[i] = lines[i]
    wnd[lines[i][0]] = [i, int(lines[i][1])]

dsets = ['pascalvoc', 'coco', 'imagenetdet', 'imagenetvid']

classes = set()

for dset in dsets:
    with open(os.path.join('datasets', 'names', dset+'_wn.names')) as f:
        lines = f.readlines()
    for l in lines:
        classes.add(l.rstrip())

print(len(classes))

paths = dict()
for cls in classes:
    if cls not in wnd.keys():
        print(cls)
        print(wn.synset_from_pos_and_offset('n', int(cls[1:]))._name)
    else:
        # print(wnd[cls])

        c = cls
        str = ''
        path = list()
        while True:
            path.append(c)
            str += wn.synset_from_pos_and_offset('n', int(c[1:]))._name + ' -> '
            if wnd[c][1] < 0:
                path.append('-1')
                path.reverse()
                paths[cls] = path
                str += 'root'
                break
            c = cats[wnd[c][1]][0]

        print(str)

tree = dict()  # values are children, keys are parents/current
for cls in classes:
    for step in paths[cls]:
        if step

from ete3 import Tree, TreeStyle
t = Tree("((((H,K)D,(F,I)G)B,E)A,((L,(N,Q)O)J,(P,S)M)C);", format=1)
# t.populate(30)
ts = TreeStyle()
ts.show_leaf_name = True
ts.mode = "c"
ts.arc_start = -360 # 0 degrees = 3 o'clock
ts.arc_span = 360
t.show(tree_style=ts)