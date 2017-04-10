
require('xlua')
torch.setdefaulttensortype('torch.FloatTensor')
paths.dofile('util.lua')
paths.dofile('loader.lua')
paths.dofile('models/model.lua')
paths.dofile('game.lua')
paths.dofile('batch.lua')
paths.dofile('train.lua')
--paths.dofile('games/init.lua')

require'gnuplot'

local cmd = torch.CmdLine()
cmd:option('--num_distractors', 1, 'the number of distractors')
cmd:option('--max_steps', 2, 'the number of distractors')

cmd:option('--shape_range', 3, '')
cmd:option('--color_range', 3, '')
cmd:option('--size_range', 2, '')

cmd:option('--cost_query', 0.05, '')
cmd:option('--cost_correct', -1, '')
cmd:option('--cost_wrong', 1, '')

cmd:option('--nchannels', 3, '')
cmd:option('--src_height', 32, '')
cmd:option('--src_width', 32, '')

cmd:option('--nonlin', 'relu', 'relu | tanh | none')
cmd:option('--inputsz', 16*5*5, '')
cmd:option('--answer_hidsz', 64, '')
cmd:option('--answer_num_symbols', 2, '')
cmd:option('--ask_num_symbols', 3, '')
cmd:option('--ask_hidsz', 64, '')

--comm
cmd:option('--comm', 'continuous', 'continuous|Gumbel|     communication mode')
cmd:option('--Gumbel_temp', 1.0, 'fixed Gumbel_temp')

-- training parameters
---------
cmd:option('--epochs', 10, 'the number of training epochs')
cmd:option('--nbatches', 10, 'the number of mini-batches in one epoch')
cmd:option('--batch_size', 32, 'size of mini-batch (the number of parallel games) in each thread')
---- GAE
cmd:option('--gamma', 0.99, 'size of mini-batch (the number of parallel games) in each thread')
cmd:option('--lambda', 0.96, 'size of mini-batch (the number of parallel games) in each thread')
---- lr
cmd:option('--lrate', 1e-3, 'learning rate')

---- baseline mixing
cmd:option('--alpha', 0.03, 'coefficient of baseline term in the cost function')
---- entropy mixing
cmd:option('--beta_start', 0, 'coefficient of listener entropy mixing')
cmd:option('--beta_end_batch', 100*50, '')

---- clipping
cmd:option('--reward_mult', 1, 'coeff to multiply reward for bprop')
cmd:option('--max_grad_norm', 0, 'gradient clip value')
cmd:option('--clip_grad', 0, 'gradient clip value')
-- for optim
cmd:option('--optim', 'rmsprop', 'optimization method: rmsprop | sgd | adam')
cmd:option('--momentum', 0, 'momentum for SGD')
cmd:option('--wdecay', 0, 'weight decay for SGD')
cmd:option('--rmsprop_alpha', 0.97, 'parameter of RMSProp')
cmd:option('--rmsprop_eps', 1e-6, 'parameter of RMSProp')
cmd:option('--adam_beta1', 0.9, 'parameter of Adam')
cmd:option('--adam_beta2', 0.999, 'parameter of Adam')
cmd:option('--adam_eps', 1e-8, 'parameter of Adam')
--other
cmd:option('--save', '', 'file name to save the model')
cmd:option('--load', 'nonlstm_speaker_at80', 'file name to load the model')


cmd:option('--init_std', 0.1, '')

g_opts = cmd:parse(arg or {})
g_init_model()
g_log = {}
train(g_opts.epochs)

