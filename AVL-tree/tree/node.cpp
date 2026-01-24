#include "include/avl_node.hpp"
#include "include/node_base.hpp"
#include "include/tree_base.hpp"

template <typename NodeType>
TreeNode<NodeType>::~TreeNode() {}

template <typename NodeType>
int TreeNode<NodeType>::get_key() {
  return key;
}

template <typename NodeType>
NodeType *TreeNode<NodeType>::get_left() {
  return left;
}

template <typename NodeType>
NodeType *TreeNode<NodeType>::get_right() {
  return right;
}

template <typename NodeType>
void TreeNode<NodeType>::set_left(NodeType *node) {
  left = node;
}

template <typename NodeType>
void TreeNode<NodeType>::set_right(NodeType *node) {
  right = node;
}

template <typename NodeType>
void TreeNode<NodeType>::set_parent(NodeType *node) {
  parent = node;
}

AvlTreeNode::AvlTreeNode(int _key, AvlTreeNode *parent) {
  parent = parent;
  key = _key;
  left = right = nullptr;
  height = 1;
}

void AvlTreeNode::fix_height() {
  if (left == nullptr) {
    height = right == nullptr ? 1 : right->height + 1;
  } else {
    height = right == nullptr ? left->height + 1
                              : std::max(right->height, left->height) + 1;
  }
}

int AvlTreeNode::balance_factor() {
  if (this == nullptr) {
    return 0;
  }

  if (left == nullptr) {
    if (right == nullptr) {
      return 0;
    }
    return -(right->height);
  }

  if (right == nullptr) {
    return left->height;
  }

  return left->height - right->height;
}

int AvlTreeNode::get_height() { return height; }

AvlTreeNode::~AvlTreeNode() {}
