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

# Show the screen
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
# import numpy as np

total_frames = 0
total_aliased_frames = 0
for episode in xrange(10):
  state = {}
  rams = {}
  total_reward = 0
  frame = 0
  aliased_screens = 0
  dims = ale.getScreenDims()
  # for a in actions:
  while not ale.game_over():
    a = legal_actions[randrange(len(legal_actions))]
    reward = ale.act(a)
    screen = tuple(ale.getScreen())
    ram = tuple(ale.getRAM())
    assert ram not in rams, 'Duplicate ram found'
    rams[ram] = None
    if screen in state:
      aliased_screens += 1
      # plt.imshow(np.array(screen).reshape((dims[1],dims[0]), order='C'))
      # plt.show()
    state[screen] = ram
    total_reward += reward
    frame += 1
  ale.reset_game()
  print 'Episode', episode, 'frames', frame, 'aliased_screens', aliased_screens
  total_frames += frame
  total_aliased_frames += aliased_screens
perc = total_aliased_frames / float(total_frames)
print 'TotalFrames', total_frames, 'TotalAliased', total_aliased_frames, 'Percentage', perc
