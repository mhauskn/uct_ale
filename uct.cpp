#include <assert.h>
#include "uct.hpp"

using namespace std;

Node::Node(ALEState state, ActionVect& possible_actions) :
    parent(NULL),
    visits(1),
    imm_reward(0),
    avg_return(0),
    state(state)
{
  untried_actions.assign(possible_actions.begin(), possible_actions.end());
}

Node::Node(Node* parent, Action a, ALEState state, float reward, ActionVect& possible_actions) :
    parent(parent),
    action(a),
    visits(1),
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

bool Node::is_leaf() {
  return children.empty();
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

// Node* Node::highest_value_child() {
//   float best_score;
//   const Node *best_child = NULL;
//   for (list<Node*>::iterator iter=children.begin(); iter!=children.end(); iter++) {
//     Node *child = *iter;
//     float score = child->value/child->visits + UCTK * sqrt(2*logf(visits)/child->visits);
//     if (best_child == NULL || score > best_score) {
//       best_score = score;
//       best_child = child;
//     }
//   }
//   return best_child;
// }

Node* Node::most_visited_child() {
  Node *best_child = NULL;
  int most_visits = 0;
  for (list<Node*>::iterator iter=children.begin(); iter!=children.end(); iter++) {
    Node *child = *iter;
    if (child->visits > most_visits) {
      best_child = child;
      most_visits = child->visits;
    }
  }
  return best_child;
}

Node* Node::add_child(Action a, ALEInterface& ale, ActionVect& possible_actions) {
  cout << "Adding new child" << endl;
  float reward = ale.act(a);
  ALEState s = ale.cloneState();
  for (list<Node*>::iterator iter=children.begin(); iter!=children.end(); iter++) {
    Node* child = *iter;
    if (child->state.equals(s)) {
      cout << "Found child match" << endl;
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

UCT::UCT(ALEInterface& ale, int search_depth, int simulations_per_step,
         float gamma, ActionVect& actions) :
    ale(ale),
    possible_actions(actions),
    search_depth(search_depth),
    simulations_per_step(simulations_per_step),
    gamma(gamma),
    root(NULL)
{
  root = new Node(ale.cloneState(), possible_actions);
}

UCT::~UCT() {
  delete root;
}

Action UCT::step() {
  for (int i=0; i<simulations_per_step; ++i) {
    sample();
    cout << "Simulation " << i << endl;
  }
  Node* most_visited_child = root->most_visited_child();
  assert(most_visited_child);
  root->prune(most_visited_child);
  delete root;
  root = most_visited_child;
  cout << "Moving to child with value: " << root->avg_return << endl;
  return most_visited_child->action;
}

void UCT::sample() {
  Node* n = root;
  int depth = 0;
  while (!n->is_leaf() && depth < search_depth) {
    if (n->untried_actions.empty()) {
      Action a = select_action(n);
      n = n->get_child(a);
    } else {
      list<Action>::iterator it = n->untried_actions.begin();
      advance(it, rand() % n->untried_actions.size());
      Action a = *it;
      n->untried_actions.erase(it);
      n = n->add_child(a, ale, possible_actions);
    }
    depth++;
  }
  float J = rollout(n, search_depth - depth);
  update_value(n, J);
}

float UCT::rollout(Node* n, int max_depth) {
  ale.restoreState(n->state);
  float total_return = 0;
  float discount = 1;
  int depth = 0;
  while (!ale.game_over() && depth < max_depth) {
    Action a = possible_actions[rand() % possible_actions.size()];
    if (depth == 0) {
      n = n->add_child(a, ale, possible_actions);
      total_return += discount * n->imm_reward;
    } else {
      total_return += discount * ale.act(a);
    }
    discount *= gamma;
    depth++;
  }
  return total_return;
}

void UCT::update_value(Node* n, float total_return) {
  float R = n->imm_reward + total_return;
  n->avg_return = n->avg_return * (n->visits / float(n->visits+1)) + R / float(n->visits+1);
  n->visits++;
  if (n->parent != NULL) {
    update_value(n->parent, gamma * R);
  }
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

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " rom_file" << std::endl;
        return 1;
    }

    int search_depth = 300;
    int simulations_per_step = 5;
    float gamma = .999;

    ALEInterface ale;
    ale.loadROM(argv[1]);
    ActionVect legal_actions = ale.getLegalActionSet();

    // ale.act(legal_actions[0]);
    // ALEState s1 = ale.cloneState();
    // for (int i=0; i<100; ++i) {
    //   ale.act(legal_actions[i%legal_actions.size()]);
    // }
    // ALEState s1_end = ale.cloneState();

    // ale.restoreState(s1);
    // ALEState s2 = ale.cloneState();
    // for (int i=0; i<100; ++i) {
    //   ale.act(legal_actions[i%legal_actions.size()]);
    // }
    // ALEState s2_end = ale.cloneState();
    // if (s1_end.equals(s2)) {
    //   cout << "Got Match" << endl;
    // } else {
    //   cout << "No Match" << endl;
    // }

    UCT uct(ale, search_depth, simulations_per_step, gamma, legal_actions);
    uct.step();

    return 0;
}
