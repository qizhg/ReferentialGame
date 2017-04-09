
torch.setdefaulttensortype('torch.FloatTensor')
--paths.dofile('util.lua')
paths.dofile('models/model.lua')
--paths.dofile('games/init.lua')

require'gnuplot'

local cmd = torch.CmdLine()
cmd:option('--num_distractors', 1, 'the number of distractors')
cmd:option('--max_steps', 2, 'the number of distractors')


cmd:option('--nonlin', 'relu', 'relu | tanh | none')
cmd:option('--inputsz', 16*5*5, '')
cmd:option('--answer_hidsz', 64, '')
cmd:option('--answer_num_symbols', 5, '')
cmd:option('--ask_num_symbols', 5, '')
cmd:option('--ask_hidsz', 64, '')




cmd:option('--init_std', 0.1, '')

g_opts = cmd:parse(arg or {})
g_init_model()
