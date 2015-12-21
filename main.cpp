#include <gflags/gflags.h>
#include <random>
#include <chrono>
#include "uct.hpp"

using namespace std;

DEFINE_int32(frame_skip, 3, "Frames skipped at each action");
DEFINE_double(repeat_action_prob, 0, "Probability to repeat actions");
DEFINE_string(rom, "", "Atari ROM file to load");
DEFINE_bool(minimal_action_set, false, "Use minimal action set");
DEFINE_int32(seed, 0, "Random seed. Default: time");

int main(int argc, char** argv) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  std::mt19937 rng(FLAGS_seed > 0 ? FLAGS_seed :
                   std::chrono::system_clock::now().time_since_epoch().count());
  ALEInterface ale;
  ale.setInt("frame_skip", FLAGS_frame_skip);
  ale.setInt("random_seed", FLAGS_seed);
  ale.setFloat("repeat_action_probability", FLAGS_repeat_action_prob);
  ale.loadROM(FLAGS_rom);
  ActionVect action_set = FLAGS_minimal_action_set ? ale.getMinimalActionSet() :
      ale.getLegalActionSet();
  ALEState start_state = ale.cloneState();
  ActionVect selected_actions;
  UCT uct(ale, action_set, rng);
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
}
