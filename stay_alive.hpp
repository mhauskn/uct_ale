#ifndef __STAY_ALIVE_HPP__
#define __STAY_ALIVE_HPP__

#include "uct.hpp"

class StayAlive : UCT {
 public:
  StayAlive(ALEInterface& ale, ActionVect& actions, std::mt19937& rng);
  std::vector<Action> run();

  bool update(Node* n);
};

#endif
