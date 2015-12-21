#include <iostream>
#include <ale_interface.hpp>
#include <list>
#include <random>

class Node {
public:
  Node(ALEState state, ActionVect& possible_actions);
  Node(Node* parent, Action a, ALEState state, float reward, ActionVect& possible_actions);
  ~Node();

  Node* highest_value_child();
  Node* most_visited_child();
  Node* add_child(Action a, ALEState s, float reward, ActionVect& possible_actions);
  Node* get_child(Action a);
  void prune(Node* chosen_child);
  inline bool has_children() { return !children.empty(); };
  inline bool is_leaf() { return children.empty(); };
  inline bool fully_expanded() { return untried_actions.empty(); };
  inline bool terminal() { return fully_expanded() && is_leaf(); };
  void print(int indent, bool print_children);

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
  UCT(ALEInterface& ale, ActionVect& actions, std::mt19937& rng);
  ~UCT();

  Action step();
  float rollout(Node* n, int depth);
  Node* rollout_add_nodes(Node* n, int depth);
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
