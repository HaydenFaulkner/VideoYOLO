### Files Information
`.tree` files are of the format `child_id  parent_id`

`9k.tree` - the original tree

`filtered.tree` - the final filtered tree (without ImageNet-DET classes)

`filtered_det.tree` - the final filtered tree (with ImageNet-DET classes)

`new_parents.tree` - the old child - parent assignments before filtering

`new_classes.txt` - replacement of class ids in format `old_id new_id` (used for merging classes across sets that are
 the same but labelled with different ids in each set)