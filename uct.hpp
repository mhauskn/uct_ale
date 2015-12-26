#ifndef __UCT_HPP__
#define __UCT_HPP__

#include <ale_interface.hpp>
#include <random>
#include "node.hpp"

class UCT {
public:
  UCT(ALEInterface& ale, ActionVect& actions, std::mt19937& rng);
  ~UCT();

  Action step();
  float rollout(Node* n, int depth);
  Node* rollout_add_nodes(Node* n, int depth);
  Node* expand(Node* n);
  void update_value(Node* n, float total_return);
  Action select_action(Node* n);
  bool game_over() { return root->terminal(); };

protected:
  ALEInterface& ale;
  ActionVect& possible_actions;
  ActionVect terminal_vec;
  std::mt19937& rng;
  Node* root;
  int time_step;
  float total_reward;
};

#endif
