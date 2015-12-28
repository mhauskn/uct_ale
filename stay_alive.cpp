#include <gflags/gflags.h>
#include <assert.h>
#include <iostream>
#include "stay_alive.hpp"

using namespace std;

DEFINE_int32(tolerance, 1000, "Consecutive backtracks allowed before quitting");

StayAlive::StayAlive(ALEInterface& ale, ActionVect& actions, mt19937& rng) :
    UCT(ale, actions, rng) {}

vector<Action> StayAlive::run() {
  Node* n = root;
  int depth = 0;
  int failed_attempts = 0;
  while (failed_attempts <= FLAGS_tolerance) {
    while (!n->fully_expanded()) {
      n = expand(n);
      depth++;
    }
    bool found_new_best = update(n);
    failed_attempts = found_new_best ? 0 : failed_attempts + 1;
    if (found_new_best) {
      cout << "NewBest: score=" << root->avg_return << " depth=" << depth << endl;
    }
    while (n->imm_reward == 0 || n->fully_expanded()) {
      if (n->parent == NULL) {
        break;
      }
      n = n->parent;
      depth--;
    }
  }
  // Return best sequence from root
  ActionVect selected_actions;
  n = root;
  while (!n->terminal()) {
    n = n->highest_value_child();
    selected_actions.push_back(n->action);
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
