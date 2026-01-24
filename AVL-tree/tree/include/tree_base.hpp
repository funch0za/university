#ifndef TREE_H
#define TREE_H

#include <string>

#include "node_base.hpp"

template <typename NodeType>
class Tree {
 protected:
  NodeType *head;

  virtual NodeType *remove_node(NodeType *node, int key) = 0;
  virtual NodeType *insert_node(NodeType *node, int key) = 0;
  virtual NodeType *remove_min(NodeType *node) = 0;

  virtual bool find(NodeType *node, int key);
  virtual NodeType *find_min(NodeType *node);

  virtual void print_graph(NodeType *node, std::string prefix);
  virtual void print_sorted(NodeType *node);

 public:
  virtual ~Tree<NodeType>();

  virtual void insert(int key) = 0;
  virtual void remove(int key) = 0;

  virtual bool find(int key);
  virtual int find_min();

  virtual bool empty();

  virtual void print_graph();
  virtual void print_sorted();
};

#endif
