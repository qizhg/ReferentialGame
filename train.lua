require 'optim'
require('nn')
require('nngraph')


function train_batch(task_id)
	local batch = batch_init(g_opts.batch_size)
	
	--referents
	local referents_src = batch_input(batch) --(#batch, 2+num_distractors, nchannels, height, width)
	local referents_src_4D = referents_src:view(-1,g_opts.nchannels,g_opts.src_height, g_opts.src_width):clone()
	local referents = preproc_model:forward(referents_src_4D)
    local referents_3D = referents:view(#batch, 2+g_opts.num_distractors,-1)
	

	--forward cache
	local ask = {}
    local ask_hidsz = g_opts.ask_hidsz
    ask.referents = referents_3D:narrow(2,1,1+g_opts.num_distractors):clone()
    ask.hid = {} 
    ask.cell = {}
    ask.hid[0] = torch.Tensor(#batch, ask_hidsz):fill(0)
    ask.cell[0] = torch.Tensor(#batch, ask_hidsz):fill(0)

    answer_msg = {}
    answer_msg[0] = torch.Tensor(#batch, g_opts.answer_num_symbols):fill(0)

    ask_msg = {}

	--forward pass
    for t = 1, g_opts.max_steps do

    	--ask
    	local ask_out = ask_model:forward( {ask.referents, answer_msg[0],ask.hid[t-1],ask.cell[t-1]} )
    	----  ask_out = {symbol_logprob, act_logprob, hidstate, cellstate}
    	ask.hid[t] = ask_out[3]:clone()
        ask.cell[t] = ask_out[4]:clone()

    end

end



function train(N)
    for n = 1, N do
		for k = 1, g_opts.nbatches do
			train_batch()
		end
    end

end