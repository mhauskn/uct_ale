#include <iostream>
#include <ale_interface.hpp>
#include <list>

class Node {
public:
  Node(ALEState state, ActionVect& possible_actions);
  Node(Node* parent, Action a, ALEState state, float reward, ActionVect& possible_actions);
  ~Node();

  // Node* highest_value_child();
  Node* most_visited_child();
  Node* add_child(Action a, ALEInterface& ale, ActionVect& possible_actions);
  Node* get_child(Action a);
  void prune(Node* chosen_child);
  bool is_leaf();

  Action action;
  Node* parent;
  std::list<Node*> children;
  int visits;
  std::list<Action> untried_actions;
  float imm_reward;
  float avg_return;
  ALEState state;
};

class UCT {
public:
  UCT(ALEInterface& ale, int search_depth, int simulations_per_step,
      float gamma, ActionVect& actions);
  ~UCT();

  Action step();
  void sample();
  float rollout(Node* n, int depth);
  void update_value(Node* n, float total_return);
  Action select_action(Node* n);

protected:
  ALEInterface& ale;
  ActionVect& possible_actions;
  int search_depth;
  int simulations_per_step;
  float gamma;
  Node* root;
};
