#include <gflags/gflags.h>
#include <assert.h>
#include "uct.hpp"

using namespace std;

DEFINE_int32(search_depth, 300, "Depth of search tree");
DEFINE_int32(simulations, 500, "Simulations to do at each UCT step");
DEFINE_double(gamma, .999, "Discount factor");
DEFINE_bool(max_uct, false, "Max vs average value in node update");

Node::Node(ALEState state, ActionVect& possible_actions) :
    parent(NULL),
    visits(0),
    imm_reward(0),
    avg_return(0),
    state(state)
{
  untried_actions.assign(possible_actions.begin(), possible_actions.end());
}

Node::Node(Node* parent, Action a, ALEState state, float reward, ActionVect& possible_actions) :
    parent(parent),
    action(a),
    visits(0),
    imm_reward(reward),
    avg_return(0),
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

Node* Node::add_child(Action a, ALEState s, float reward, ActionVect& possible_actions) {
  for (list<Node*>::iterator iter=children.begin(); iter!=children.end(); iter++) {
    Node* child = *iter;
    if (child->state.equals(s)) {
      child->visits--; // Offset unintentional visit
      return child;
    }
  }
  Node* child = new Node(this, a, s, reward, possible_actions);
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
       << " v=" << visits << endl;
  if (print_children) {
    for (list<Node*>::iterator iter=children.begin(); iter!=children.end(); iter++) {
      Node* child = *iter;
      child->print(indent+2, print_children);
    }
  }
}

UCT::UCT(ALEInterface& ale, ActionVect& actions, mt19937& rng) :
    ale(ale),
    possible_actions(actions),
    root(NULL),
    time_step(0),
    total_reward(0),
    rng(rng)
{
  root = new Node(ale.cloneState(), possible_actions);
}

UCT::~UCT() {
  delete root;
}

Action UCT::step() {
  for (int i=0; i<FLAGS_simulations; ++i) {
    Node* n = root;
    int depth = 0;
    // Select
    while (n->fully_expanded() && n->has_children() && depth++ < FLAGS_search_depth) {
      Action a = select_action(n);
      n = n->get_child(a);
    }
    // Expand
    if (!n->fully_expanded()) {
      list<Action>::iterator it = n->untried_actions.begin();
      advance(it, rng() % n->untried_actions.size());
      Action a = *it;
      n->untried_actions.erase(it);
      ale.restoreState(n->state);
      float reward = ale.act(a);
      if (ale.game_over()) {
        n = n->add_child(a, ale.cloneState(), reward, terminal_vec);
      } else {
        n = n->add_child(a, ale.cloneState(), reward, possible_actions);
      }
      depth++;
    }
    if (FLAGS_max_uct) {
      n = rollout_add_nodes(n, FLAGS_search_depth - depth);
      update_value(n, 0);
    } else {
      float J = rollout(n, FLAGS_search_depth - depth);
      update_value(n, J);
    }
  }
  Node* selected_child = root->highest_value_child(); //root->most_visited_child();
  assert(selected_child);
  root->prune(selected_child);
  delete root;
  selected_child->parent = NULL;
  root = selected_child;
  total_reward += selected_child->imm_reward;
  cout << "t=" << time_step
       << " a=" << selected_child->action
       << " r=" << selected_child->imm_reward
       << " J=" << selected_child->avg_return
       << " R=" << total_reward
       << endl;
  time_step++;
  return selected_child->action;
}

float UCT::rollout(Node* n, int max_depth) {
  ale.restoreState(n->state);
  float total_return = 0;
  float discount = 1;
  int depth = 0;
  while (!ale.game_over() && depth < max_depth) {
    Action a = possible_actions[rng() % possible_actions.size()];
    float reward = ale.act(a);
    total_return += discount * reward;
    discount *= FLAGS_gamma;
    depth++;
  }
  return total_return;
}

Node* UCT::rollout_add_nodes(Node* n, int max_depth) {
  ale.restoreState(n->state);
  int depth = 0;
  while (!ale.game_over() && depth < max_depth) {
    Action a = possible_actions[rng() % possible_actions.size()];
    float reward = ale.act(a);
    depth++;
    n = n->add_child(a, ale.cloneState(), reward, possible_actions);
  }
  return n;
}

void UCT::update_value(Node* n, float total_return) {
  float R = n->imm_reward + total_return;
  do {
    if (FLAGS_max_uct) {
      n->avg_return = max(R, n->avg_return);
    } else {
      n->avg_return = n->avg_return * (n->visits / float(n->visits+1))
          + R / float(n->visits+1);
    }
    n->visits++;
    R *= FLAGS_gamma;
    n = n->parent;
  } while (n != NULL);
}

Action UCT::select_action(Node* n) {
  float best_score;
  const Node *best_child = NULL;
  for (list<Node*>::iterator iter=n->children.begin(); iter!=n->children.end(); iter++) {
    Node *child = *iter;
    float score = child->avg_return + sqrt(logf(child->visits)/n->visits);
    if (best_child == NULL || score > best_score) {
      best_score = score;
      best_child = child;
    }
  }
  return best_child->action;
}
