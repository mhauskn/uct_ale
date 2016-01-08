#include <gflags/gflags.h>
#include <random>
#include <chrono>
#include <SDL.h>
#include "stay_alive.hpp"

using namespace std;

DEFINE_int32(frame_skip, 3, "Frames skipped at each action");
DEFINE_double(repeat_action_prob, 0, "Probability to repeat actions");
DEFINE_string(rom, "", "Atari ROM file to load");
DEFINE_bool(minimal_action_set, false, "Use minimal action set");
DEFINE_int32(seed, 0, "Random seed. Default: time");
DEFINE_bool(display, false, "Visualize the policy");

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
  StayAlive alive(ale, action_set, rng);
  selected_actions = alive.run();

  cout << "ActionSequence: ";
  float total_reward = 0;

  if (FLAGS_display) {
    ale.setBool("display_screen", true);
    ale.setBool("sound", true);
    ale.loadROM(FLAGS_rom);
  }
  ale.reset_game();
  // ale.restoreState(start_state);

  for (int i=0; i<selected_actions.size(); ++i) {
    cout << selected_actions[i] << " ";
    total_reward += ale.act(selected_actions[i]);
  }
  cout << endl;
  cout << "Total Reward: " << total_reward << endl;
}
