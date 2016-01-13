#!/usr/bin/env python
import sys
from random import randrange
from ale_python_interface import ALEInterface

if len(sys.argv) < 4:
  print 'Usage:', sys.argv[0], 'OutputFile ROMFile recordDir'
  sys.exit()

out_file = sys.argv[1]
rom = sys.argv[2]
record_dir = sys.argv[3]

# Parse the actions
def parse_log_file(fname):
  with open(fname) as f:
    for line in f:
      if line.startswith('ActionSequence: '):
        f.close()
        return map(int, line.split(' ')[1:-1])

actions = parse_log_file(out_file)

ale = ALEInterface()
ale.setInt("frame_skip", 3);
ale.setFloat("repeat_action_probability", 0);
ale.setInt('random_seed', 123)
ale.setBool('sound', True)
ale.setBool('display_screen', True)
ale.setString('record_screen_dir', record_dir);
ale.setString('record_sound_filename', record_dir + '/sound.wav');

ale.loadROM(rom)
total_reward = 0
for a in actions:
  reward = ale.act(a);
  total_reward += reward

print 'Total score:', total_reward
