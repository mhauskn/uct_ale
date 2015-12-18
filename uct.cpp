#include <assert.h>
#include "uct.hpp"

using namespace std;

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
  cout << s << "a=" << action << " r=" << imm_reward << " J=" << avg_return << " v=" << visits << endl;
  if (print_children) {
    for (list<Node*>::iterator iter=children.begin(); iter!=children.end(); iter++) {
      Node* child = *iter;
      child->print(indent+2, print_children);
    }
  }
}

UCT::UCT(ALEInterface& ale, int search_depth, int simulations_per_step,
         float gamma, ActionVect& actions) :
    ale(ale),
    possible_actions(actions),
    search_depth(search_depth),
    simulations_per_step(simulations_per_step),
    gamma(gamma),
    root(NULL),
    time_step(0)
{
  root = new Node(ale.cloneState(), possible_actions);
}

UCT::~UCT() {
  delete root;
}

Action UCT::step() {
  for (int i=0; i<simulations_per_step; ++i) {
    Node* n = root;
    int depth = 0;
    // Select
    while (n->fully_expanded() && n->has_children() && depth++ < search_depth) {
      Action a = select_action(n);
      n = n->get_child(a);
    }
    // Expand
    if (!n->fully_expanded()) {
      list<Action>::iterator it = n->untried_actions.begin();
      advance(it, rand() % n->untried_actions.size());
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
    // Rollout
    float J = rollout(n, search_depth - depth);
    // Backpropagate
    update_value(n, J);
  }
  Node* selected_child = root->highest_value_child(); //root->most_visited_child();
  assert(selected_child);
  root->prune(selected_child);
  delete root;
  selected_child->parent = NULL;
  root = selected_child;
  cout << "t=" << time_step
       << " a=" << selected_child->action
       << " r=" << selected_child->imm_reward
       << " J=" << selected_child->avg_return << endl;
  time_step++;
  return selected_child->action;
}

float UCT::rollout(Node* n, int max_depth) {
  ale.restoreState(n->state);
  float total_return = 0;
  float discount = 1;
  int depth = 0;
  while (!ale.game_over() && depth < max_depth) {
    Action a = possible_actions[rand() % possible_actions.size()];
    total_return += discount * ale.act(a);
    discount *= gamma;
    depth++;
  }
  return total_return;
}

void UCT::update_value(Node* n, float total_return) {
  float R = n->imm_reward + total_return;
  do {
    n->avg_return = n->avg_return * (n->visits / float(n->visits+1)) + R / float(n->visits+1);
    n->visits++;
    R *= gamma;
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

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " rom_file" << std::endl;
        return 1;
    }

    int search_depth = 300;
    int simulations_per_step = 10;
    float gamma = .999;
    int frame_skip = 40;
    float repeat_action_prob = 0;

    ALEInterface ale;
    ale.setInt("frame_skip", frame_skip);
    ale.setFloat("repeat_action_probability", repeat_action_prob);
    ale.loadROM(argv[1]);
    ActionVect legal_actions = ale.getLegalActionSet();

    ALEState start_state = ale.cloneState();
    ActionVect selected_actions;
    UCT uct(ale, search_depth, simulations_per_step, gamma, legal_actions);
    while (!uct.game_over()) {
      Action a = uct.step();
      selected_actions.push_back(a);
    }

    cout << "Replaying UCT Actions" << endl;
    float total_reward = 0;
    ale.restoreState(start_state);
    for (int i=0; i<selected_actions.size(); ++i) {
      assert(!ale.game_over());
      total_reward += ale.act(selected_actions[i]);
    }
    assert(ale.game_over());
    cout << "Total Reward: " << total_reward << endl;
    return 0;
}
