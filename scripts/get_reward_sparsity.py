#!/usr/bin/env python
import sys
from random import randrange
from ale_python_interface import ALEInterface

if len(sys.argv) < 2:
  print 'Usage:', sys.argv[0], 'rom'
  sys.exit()

rom = sys.argv[1]

# out_file = sys.argv[1]
# Parse the actions
# def parse_log_file(fname):
#   with open(fname) as f:
#     for line in f:
#       if line.startswith('ActionSequence: '):
#         f.close()
#         return map(int, line.split(' ')[1:-1])

# actions = parse_log_file(out_file)
# print 'Action Sequence Length:', len(actions)

ale = ALEInterface()
ale.setInt("frame_skip", 3);
ale.setFloat("repeat_action_probability", 0);
ale.setInt('random_seed', 123)
# ale.setBool('sound', True)
# ale.setBool('display_screen', True)

ale.loadROM(rom)
legal_actions = ale.getLegalActionSet()

total_frames = 0
pos_reward_frames, neg_reward_frames = 0, 0
for episode in xrange(100):
  total_reward = 0
  frame = 0
  dims = ale.getScreenDims()
  # for a in actions:
  while not ale.game_over():
    a = legal_actions[randrange(len(legal_actions))]
    reward = ale.act(a)
    if reward > 0:
      pos_reward_frames += 1
    if reward < 0:
      neg_reward_frames += 1
    total_reward += reward
    frame += 1
  ale.reset_game()
  print 'Episode', episode, 'frames', frame
  total_frames += frame
pos_perc = pos_reward_frames / float(total_frames)
neg_perc = neg_reward_frames / float(total_frames)
print 'TotalFrames', total_frames, 'PosRewardFrames', pos_reward_frames,\
  'NegRewardFrames', neg_reward_frames, 'PosPercentage', pos_perc,\
  'NegPercentage', neg_perc
