#include <gflags/gflags.h>
#include <assert.h>
#include <iostream>
#include "stay_alive.hpp"

using namespace std;

DEFINE_int32(tolerance, 1000, "Consecutive backtracks allowed before quitting");
DEFINE_int32(max_depth, 1000000, "Quit after trajectory has reached this depth");

StayAlive::StayAlive(ALEInterface& ale, ActionVect& actions, mt19937& rng) :
    UCT(ale, actions, rng) {}

vector<Action> StayAlive::run() {
  vector<Action> selected_actions;
  Node* n = root;
  while (!n->terminal() && selected_actions.size() < FLAGS_max_depth) {
    vector<Action> chosen_actions = explore_until_mistake(n);
    for (int i=0; i<chosen_actions.size(); ++i) {
      Action a = chosen_actions[i];
      selected_actions.push_back(a);
      Node* chosen = n->get_child(a);
      n->prune(chosen);
      n = chosen;
    }
  }
  assert(n->terminal());
  return selected_actions;
}

vector<Action> StayAlive::explore_until_mistake(Node* start_node) {
  Node* n = start_node;
  int depth = 0;
  int failed_attempts = 0;
  while (failed_attempts < FLAGS_tolerance && depth < FLAGS_max_depth) {
    while (!n->fully_expanded()) {
      n = expand(n);
      depth++;
      if (n->terminal() || n->lost_life() || n->imm_reward < 0) {
        break;
      }
    }
    bool found_new_best = update(n);
    if (found_new_best) {
      cout << "NewBest: score=" << start_node->avg_return << " depth=" << depth
           << " failed_attempts=" << failed_attempts << endl;
    }
    failed_attempts = found_new_best ? 0 : failed_attempts + 1;
    // Go up to last postive reward (or start_node)
    while (n->imm_reward <= 0 || n->fully_expanded()) {
      if (n == start_node || n->parent == NULL) {
        break;
      }
      n = n->parent;
      depth--;
    }
    assert(n->imm_reward > 0 || n == start_node);
    // Go down randomly until we hit a non-fully expanded node
    while (n->fully_expanded() && n->has_children()) {
      n = n->random_child(rng);
      depth++;
    }
    assert(!n->fully_expanded() || !n->has_children());
  }
  // Return best action sequence from start_node to terminal
  ActionVect selected_actions;
  n = start_node;
  depth = 0;
  float total_score = 0;
  while (n->has_children()) {
    n = n->highest_value_child();
    selected_actions.push_back(n->action);
    depth++;
    total_score += n->imm_reward;
  }
  cout << "ExhaustedTolerance=" << failed_attempts
       << " depth=" << depth << " score=" << total_score
       << " result=";
  if (n->terminal()) {
    cout << "GameOver"<< endl;
  } else if (n->lost_life()) {
    cout << "LostLife Lives=" << n->lives << endl;
  } else if (n->imm_reward < 0) {
    cout << "NegReward=" << n->imm_reward << endl;
  }
  return selected_actions;
}

bool StayAlive::update(Node* n) {
  float R = 0;
  do {
    R += n->imm_reward;
    if (n->avg_return >= R) {
      return false;
    }
    n->avg_return = R;
    n = n->parent;
  } while (n != NULL);
  return true;
}
