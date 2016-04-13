#include <gflags/gflags.h>
#include <assert.h>
#include "uct.hpp"

using namespace std;

DEFINE_int32(search_depth, 30, "Depth of search tree");
DEFINE_int32(simulations, 50, "Simulations to do at each UCT step");
DEFINE_double(gamma, .999, "Discount factor");
DEFINE_bool(max_uct, false, "Max vs average value in node update");

UCT::UCT(ALEInterface& ale, ActionVect& actions, mt19937& rng) :
    ale(ale),
    possible_actions(actions),
    root(NULL),
    time_step(0),
    total_reward(0),
    rng(rng)
{
  root = new Node(ale.cloneState(), ale.lives(), possible_actions);
}

UCT::~UCT() {
  delete root;
}

Node* UCT::expand(Node* n) {
  if (!n->fully_expanded()) {
    list<Action>::iterator it = n->untried_actions.begin();
    advance(it, rng() % n->untried_actions.size());
    Action a = *it;
    n->untried_actions.erase(it);
    ale.restoreState(n->state);
    float reward = ale.act(a);
    int lives = ale.lives();
    if (ale.game_over()) {
      n = n->add_child(a, ale.cloneState(), reward, lives, terminal_vec);
    } else {
      n = n->add_child(a, ale.cloneState(), reward, lives, possible_actions);
    }
  }
  return n;
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
    n = expand(n);
    depth++;
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
  cout << "timestep=" << time_step
       << ", action=" << selected_child->action
       << ", reward=" << selected_child->imm_reward
       << ", expectedReturn=" << selected_child->avg_return
       << ", cumulativeReward=" << total_reward
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
    int lives = ale.lives();
    depth++;
    n = n->add_child(a, ale.cloneState(), reward, lives, possible_actions);
  }
  return n;
}

void UCT::update_value(Node* n, float total_return) {
  float R = total_return;
  do {
    R += n->imm_reward;
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
