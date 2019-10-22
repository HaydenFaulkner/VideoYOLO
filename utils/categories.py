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

# create the paths from root to leaf nodes as dict of lists keyed on leaf node wid
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

# manually done with imagenet.stanford.edu/synset?wnid=n02870092 and imagenet.stanford.edu/synset?wnid=n03346898
paths['n02870092'] = ['-1', 'n00002684', 'n00003553', 'n00021939', 'n03129123', 'n04007894', 'n02870092']
paths['n03346898'] = ['-1', 'n00002684', 'n00003553', 'n00021939', 'n04564698', 'n03895293', 'n03089014', 'n04493505',
                      'n03944672', 'n03206158', 'n03550916', 'n03346898']

# build some more structure dicts
tree = dict()  # values are children, keys are parents/current
lvls = dict()  # keys are levels from root, and values are wids
parents = dict()  # keys are wids and values are parent wids
for cls in classes:
    sub_tree = tree
    for i in range(len(paths[cls])):
        if paths[cls][i] not in sub_tree:
            sub_tree[paths[cls][i]] = dict()

            if i not in lvls:
                lvls[i] = [paths[cls][i]]
            else:
                lvls[i].append(paths[cls][i])

            if i > 0:
                if paths[cls][i] not in parents:
                    parents[paths[cls][i]] = paths[cls][i-1]
                else:
                    assert parents[paths[cls][i]] == paths[cls][i-1]
        sub_tree = sub_tree[paths[cls][i]]  # go a step deeper


# lets build and write out a .tree file
ordered_list = list()
for l in sorted(list(lvls.keys()))[1:]:
    for id in lvls[l]:
        ordered_list.append([id, parents[id]])

with open(os.path.join('datasets', 'names', 'mini.tree'), 'w') as f:
    for id in ordered_list:
        f.write('\t'.join(id)+'\n')

# let's make some pretty graphs
from anytree import Node, RenderTree

nodes = dict()
nodes['-1'] = Node('ROOT')
for k, v in parents.items():
    nodes[k] = Node(k, nodes[v])

for pre, fill, node in RenderTree(nodes['-1']):
    print("%s%s" % (pre, node.name))

with open(os.path.join('datasets', 'names', 'mini_wn.treevis'), 'w') as f:
    for pre, fill, node in RenderTree(nodes['-1']):
        f.write("%s%s\n" % (pre, node.name))

with open(os.path.join('datasets', 'names', 'mini.treevis'), 'w') as f:
    for pre, fill, node in RenderTree(nodes['-1']):
        if node.name =='ROOT':
            name = 'ROOT'
        else:
            name = wn.synset_from_pos_and_offset('n', int(node.name[1:]))._name
        f.write("%s%s\n" % (pre, name))

# from ete3 import Tree, TreeStyle
# t = Tree("((((H,K)D,(F,I)G)B,E)A,((L,(N,Q)O)J,(P,S)M)C);", format=1)
# # t.populate(30)
# ts = TreeStyle()
# ts.show_leaf_name = True
# ts.mode = "c"
# ts.arc_start = -360 # 0 degrees = 3 o'clock
# ts.arc_span = 360
# t.show(tree_style=ts)