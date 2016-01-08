#include <iostream>
#include "node.hpp"

using namespace std;

Node::Node(ALEState state, int lives, ActionVect& possible_actions) :
    parent(NULL),
    visits(0),
    imm_reward(0),
    avg_return(-1e10),
    lives(lives),
    state(state)
{
  untried_actions.assign(possible_actions.begin(), possible_actions.end());
}

Node::Node(Node* parent, Action a, ALEState state, float reward, int lives,
           ActionVect& possible_actions) :
    parent(parent),
    action(a),
    visits(0),
    imm_reward(reward),
    avg_return(-1e10),
    lives(lives),
    state(state)
{
  untried_actions.assign(possible_actions.begin(), possible_actions.end());
}

Node::~Node() {
  for (list<Node*>::iterator iter=children.begin(); iter!=children.end(); iter++) {
    Node* child = *iter;
    delete child;
  }
}

Node* Node::get_child(Action a) {
  for (list<Node*>::iterator iter=children.begin(); iter!=children.end(); iter++) {
    Node *child = *iter;
    if (child->action == a) {
      return child;
    }
  }
  return NULL;
}

Node* Node::highest_value_child() {
  float best_return;
  Node *best_child = NULL;
  for (list<Node*>::iterator iter=children.begin(); iter!=children.end(); iter++) {
    Node *child = *iter;
    if (best_child == NULL || child->avg_return > best_return) {
      best_return = child->avg_return;
      best_child = child;
    }
  }
  return best_child;
}

Node* Node::most_visited_child() {
  Node *best_child = NULL;
  int most_visits = 0;
  float best_return;
  for (list<Node*>::iterator iter=children.begin(); iter!=children.end(); iter++) {
    Node *child = *iter;
    if (child->visits > most_visits) {
      best_child = child;
      most_visits = child->visits;
      best_return = child->avg_return;
    } else if (child->visits == most_visits) {
      if (child->avg_return > best_return) {
        best_child = child;
        best_return = child->avg_return;
      }
    }
  }
  return best_child;
}

Node* Node::random_child(std::mt19937& rng) {
  list<Node*>::iterator it = children.begin();
  advance(it, rng() % children.size());
  return *it;
}

Node* Node::add_child(Action a, ALEState s, float reward, int lives,
                      ActionVect& possible_actions) {
  for (list<Node*>::iterator iter=children.begin(); iter!=children.end(); iter++) {
    Node* child = *iter;
    if (child->state.equals(s)) {
      child->visits--; // Offset unintentional visit
      return child;
    }
  }
  Node* child = new Node(this, a, s, reward, lives, possible_actions);
  children.push_back(child);
  return child;
}

void Node::prune(Node* chosen_child) {
  for (list<Node*>::iterator iter=children.begin(); iter!=children.end(); iter++) {
    Node* child = *iter;
    if (child != chosen_child) {
      delete child;
    }
  }
  children.clear();
}

void Node::print(int indent, bool print_children) {
  string s;
  for (int i=0; i<indent; ++i) { s += " "; };
  cout << s << "a=" << action << " r=" << imm_reward << " J=" << avg_return
       << " v=" << visits << " untried=" << untried_actions.size()
       << " children=" << children.size() << endl;
  if (print_children) {
    for (list<Node*>::iterator iter=children.begin(); iter!=children.end(); iter++) {
      Node* child = *iter;
      child->print(indent+2, print_children);
    }
  }
}
