"""
Code for building our WordTree
"""
import os
import treeswift  # used for going from noded tree to newick

from anytree import Node, RenderTree, LevelOrderIter  # initial tree def with console printing
from ete3 import Tree, TreeStyle, NodeStyle, AttrFace, faces  # graphing with newick representation
from nltk.corpus import wordnet as wn


# helper to go from n00000000 id to name
def id_to_name(id):
    return wn.synset_from_pos_and_offset('n', int(id[1:]))._name


with open(os.path.join('datasets', 'trees', '9k.tree')) as f:
    lines = f.readlines()
lines = [l.rstrip().split() for l in lines]

cats = dict()
wnd = dict()
inds = dict()
for i in range(len(lines)):
    cats[i] = lines[i]
    wnd[lines[i][0]] = [i, int(lines[i][1])]

dsets = ['pascalvoc', 'coco', 'imagenetdet', 'imagenetvid']
# dsets = ['pascalvoc', 'coco', 'imagenetvid']

out_str = ''
if 'imagenetdet' in dsets:
    out_str = '_det'

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
        print(id_to_name(cls))
    else:
        # print(wnd[cls])

        c = cls
        str = ''
        path = list()
        while True:
            path.append(c)
            str += id_to_name(c) + ' -> '
            if wnd[c][1] < 0:
                path.append('ROOT')
                path.reverse()
                paths[cls] = path
                str += 'ROOT'
                break
            c = cats[wnd[c][1]][0]

        print(str)

# manually done with imagenet.stanford.edu/synset?wnid=n02870092 and imagenet.stanford.edu/synset?wnid=n03346898
paths['n02870092'] = ['ROOT', 'n00002684', 'n00003553', 'n00021939', 'n03129123', 'n04007894', 'n02870092']
paths['n03346898'] = ['ROOT', 'n00002684', 'n00003553', 'n00021939', 'n04564698', 'n03895293', 'n03089014', 'n04493505',
                      'n03944672', 'n03206158', 'n03550916', 'n03346898']


# build some more structure dicts
tree = dict()  # values are children, keys are parents/current
parents = dict()  # keys are wids and values are parent wids
for cls in classes:
    sub_tree = tree
    for i in range(len(paths[cls])):
        if paths[cls][i] not in sub_tree:
            sub_tree[paths[cls][i]] = dict()

            if i > 0:
                if paths[cls][i] not in parents:
                    parents[paths[cls][i]] = paths[cls][i-1]
                else:
                    assert parents[paths[cls][i]] == paths[cls][i-1]
        sub_tree = sub_tree[paths[cls][i]]  # go a step deeper

# let's make some pretty graphs
nodes = dict()
nodes['ROOT'] = Node('ROOT')
for k, v in parents.items():
    nodes[k] = Node(k, nodes[v])

for pre, fill, node in RenderTree(nodes['ROOT']):
    print("%s%s" % (pre, node.name))

with open(os.path.join('datasets', 'trees', 'mini_wn'+out_str+'.treevis'), 'w') as f:
    for pre, fill, node in RenderTree(nodes['ROOT']):
        f.write("%s%s\n" % (pre, node.name))

with open(os.path.join('datasets', 'trees', 'mini'+out_str+'.treevis'), 'w') as f:
    for pre, fill, node in RenderTree(nodes['ROOT']):
        if node.name =='ROOT':
            name = 'ROOT'
        else:
            name = id_to_name(node.name)
        f.write("%s%s\n" % (pre, name))

# lets build and write out a .tree file
with open(os.path.join('datasets', 'trees', 'mini'+out_str+'.tree'), 'w') as f:
    for node in LevelOrderIter(nodes['ROOT']):
        if node.name != 'ROOT':
            f.write(node.name + '\t' + parents[node.name] + '\n')

# build filtered tree
# manual overwrite of parents
with open(os.path.join('datasets', 'trees', 'new_parents.tree')) as f:
    lines = f.readlines()
new_parents = [l.rstrip().split() for l in lines]

for np in new_parents:
    parents[np[0]] = np[1]

# manual deletion of wn_ids
with open(os.path.join('datasets', 'trees', 'removed_wn.tree')) as f:
    lines = f.readlines()
remove_wn = [l.rstrip() for l in lines]

for wnid in remove_wn:
    for c, p in parents.items():  # for all items with the parent we are about to delete
        if p == wnid:
            parents[c] = parents[p]  # give it the parent of the parent
    if wnid in parents:
        del parents[wnid]  # once all assigned to grandparents we can delete


# let's make some pretty graphs
nodes = dict()
nodes['ROOT'] = Node('ROOT')
for c, p in parents.items():  # init the nodes all to root
    nodes[c] = Node(c, nodes['ROOT'])

for c, p in parents.items():  # change their parents
    nodes[c].parent = nodes[p]

for pre, fill, node in RenderTree(nodes['ROOT']):
    print("%s%s" % (pre, node.name))

with open(os.path.join('datasets', 'trees', 'filtered_wn'+out_str+'.treevis'), 'w') as f:
    for pre, fill, node in RenderTree(nodes['ROOT']):
        f.write("%s%s\n" % (pre, node.name))

with open(os.path.join('datasets', 'trees', 'filtered'+out_str+'.treevis'), 'w') as f:
    for pre, fill, node in RenderTree(nodes['ROOT']):
        if node.name =='ROOT':
            name = 'ROOT'
        else:
            name = id_to_name(node.name)
        f.write("%s%s\n" % (pre, name))

# lets build and write out a .tree file
with open(os.path.join('datasets', 'trees', 'filtered'+out_str+'.tree'), 'w') as f:
    for node in LevelOrderIter(nodes['ROOT']):
        if node.name != 'ROOT':
            f.write(node.name + '\t' + parents[node.name]+'\n')

# do some wicked graphing
nodes2 = dict()
nodes2['ROOT'] = treeswift.Node(label='ROOT')
for c, p in parents.items():  # init the nodes all to root
    nodes2[c] = treeswift.Node(label=id_to_name(c))

for c, p in parents.items():  # change their parents
    nodes2[c].set_parent(nodes2[p])
    nodes2[p].add_child(nodes2[c])

# organise children by number of sub children so it graphs prettier
for p in set(parents.values()):
    children = [[c, c.num_children()] for c in nodes2[p].child_nodes()]
    children = sorted(children, key=lambda x: x[1]) # sort on number
    for c, _ in children:
        nodes2[p].remove_child(c)
    for c, _ in children:
        nodes2[p].add_child(c)

print(nodes2['ROOT'].is_root())
print(nodes2['ROOT'].child_nodes())
print(nodes2['ROOT'].is_leaf())
print(nodes2['ROOT'].newick())

def my_layout(node):
    if node.is_leaf():
        # If terminal node, draws its name
        name_face = AttrFace("name", ftype="Roboto", fgcolor='#191919')
        name_face.margin_right = 10
        name_face.margin_left = 10
    else:
        # If internal node, draws label with smaller font size
        name_face = AttrFace("name", fsize=10, ftype="Roboto", fgcolor='#191919')
        name_face.margin_right = 10
        name_face.margin_left = 10
    # Adds the name face to the image at the preferred position
    faces.add_face_to_node(name_face, node, column=0, position="branch-right")


t = Tree(nodes2['ROOT'].newick()+';', format=8)
ts = TreeStyle()
ts.mode = "c"
ts.arc_start = 0
ts.arc_span = 360
ts.force_topology = True
ts.show_scale = False
ts.branch_vertical_margin = 2
ts.root_opening_factor = 0


# Do not add leaf trees automatically
ts.show_leaf_name = False
# Use my custom layout
ts.layout_fn = my_layout
nstyle = NodeStyle()
nstyle["size"] = 0
for n in t.traverse():
   n.set_style(nstyle)


c = t&"artifact.n.01"
cn = NodeStyle()
cn["size"] = 0
c.set_style(cn)
c.img_style["bgcolor"] = "#e9f5ff"

c = t&"living_thing.n.01"
cn = NodeStyle()
cn["size"] = 0
c.set_style(cn)
c.img_style["bgcolor"] = "#f3f3f3"

t.render(os.path.join('datasets', 'trees', 'filtered_tree'+out_str+'.pdf'), w=1000, units="mm", tree_style=ts)
t.render(os.path.join('datasets', 'trees', 'filtered_tree'+out_str+'.png'), w=1000, units="mm", tree_style=ts)
t.show(tree_style=ts)