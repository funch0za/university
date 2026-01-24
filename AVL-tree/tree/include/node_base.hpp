#ifndef TREE_NODE_H
#define TREE_NODE_H

#include <iostream>

template <typename NodeType>
class TreeNode {
 protected:
  int key;
  NodeType *left, *right, *parent;

 public:
  virtual ~TreeNode<NodeType>();
  virtual int get_key();
  virtual NodeType *get_left();
  virtual NodeType *get_right();
  virtual void set_left(NodeType *node);
  virtual void set_right(NodeType *node);
  virtual void set_parent(NodeType *node);
};

#endif
