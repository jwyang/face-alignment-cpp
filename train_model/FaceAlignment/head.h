#ifndef HEAD_H
#define HEAD_H

typedef struct sHead{
	int __num_stage;
	int __num_point;
	int __num_tree_per_point;
	int __tree_depth;
	int __node_step;

	int __num_node;
	int __num_leaf;
	int __dim_tree;

	int __num_tree_per_stage;
	int __num_tree_total;
	int __dim_feat;
}sHead;

#endif