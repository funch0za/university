#ifndef AVL_TREE_H
#define AVL_TREE_H

#include <iostream>

#include "avl_node.hpp"
#include "tree_base.hpp"

class AvlTree final : public Tree<AvlTreeNode> {
 public:
  AvlTree();
  ~AvlTree();

  void insert(int key) override;
  void remove(int key) override;

 private:
  AvlTreeNode *balance(AvlTreeNode *node);

  AvlTreeNode *rotate_right(AvlTreeNode *node);
  AvlTreeNode *rotate_left(AvlTreeNode *node);

  AvlTreeNode *insert_node(AvlTreeNode *node, int key) override;
  AvlTreeNode *remove_node(AvlTreeNode *node, int key) override;
  AvlTreeNode *remove_min(AvlTreeNode *node) override;
};

#endif
