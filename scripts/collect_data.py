#!/usr/bin/env python
import sys
import numpy as np
from random import randrange
from ale_python_interface import ALEInterface
import h5py

if len(sys.argv) < 4:
  print 'Usage:', sys.argv[0], 'ROM DatasetSize SaveFile'
  sys.exit()

rom = sys.argv[1]
dataset_size = int(sys.argv[2])
save_file = sys.argv[3]
COMPRESSION=4
BUF=65536
h5f = h5py.File(save_file, 'w')

ale = ALEInterface()
ale.setInt("frame_skip", 3);
ale.setFloat("repeat_action_probability", 0);
ale.setInt('random_seed', 123)

ale.loadROM(rom)
legal_actions = ale.getLegalActionSet()
dims = ale.getScreenDims()
collected_screens = 0
screen_dset = h5f.create_dataset('screens', (dataset_size, dims[1], dims[0], 1),
                                 dtype='uint8', compression='gzip',
                                 compression_opts=COMPRESSION)
ram_dset = h5f.create_dataset('ram', (dataset_size, 128), dtype='uint8',
                              compression='gzip', compression_opts=COMPRESSION)
tmp_screens = np.zeros([BUF, dims[1], dims[0], 1], dtype='uint8')
tmp_ram = np.zeros([BUF, 128], dtype='uint8')
while collected_screens < dataset_size:
  while collected_screens < dataset_size and not ale.game_over():
    a = legal_actions[randrange(len(legal_actions))]
    reward = ale.act(a)
    screen = ale.getScreen()
    ram = ale.getRAM()
    tmp_screens[collected_screens % BUF,:,:,0] = screen.reshape(dims[1], dims[0])
    tmp_ram[collected_screens % BUF] = ram[:]
    if collected_screens > 0 and collected_screens % BUF == 0:
      print('Writing tmp buffer...')
      screen_dset[(collected_screens - BUF) : collected_screens] = tmp_screens
      ram_dset[(collected_screens - BUF) : collected_screens] = tmp_ram
      print('Done!')
    collected_screens += 1
    if collected_screens % 100 == 0:
      print 'Collected:', collected_screens
      sys.stdout.flush()
  ale.reset_game()
print('Writing final segment...')
extra = collected_screens % BUF
screen_dset[-extra:] = tmp_screens[:extra]
ram_dset[-extra:] = tmp_ram[:extra]
h5f.close()
print('Done!')
