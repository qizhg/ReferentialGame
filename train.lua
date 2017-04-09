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
    active = {}
    action = {}
    reward = {}
    comm_mask = {}
    comm_mask[0] = torch.Tensor(#batch, 1):fill(1)

    ----ask
	local ask = {}
    local ask_hidsz = g_opts.ask_hidsz
    ask.referents = referents_3D:narrow(2,1,1+g_opts.num_distractors):clone()
    ask.comm_in = {}
    ask.comm_out={}
    ask.hid = {} 
    ask.cell = {}
    ask.hid[0] = torch.Tensor(#batch, ask_hidsz):fill(0)
    ask.cell[0] = torch.Tensor(#batch, ask_hidsz):fill(0)
    ask.baseline = {}

    ----answer
    local answer = {}
    local answer_hidsz = g_opts.answer_hidsz
    answer.target = referents_3D:narrow(2,2+g_opts.num_distractors,1):squeeze():clone()
    answer.comm_in = {}
    answer.comm_out={}
    answer.comm_out[0] = torch.Tensor(#batch, g_opts.answer_num_symbols):fill(0)
    answer.hid = {} 
    answer.cell = {}
    answer.hid[0] = torch.Tensor(#batch, answer_hidsz):fill(0)
    answer.cell[0] = torch.Tensor(#batch, answer_hidsz):fill(0)

    

	--forward pass
    for t = 1, g_opts.max_steps do
        active[t] = batch_active(batch)

    	--ask
        ask.comm_in[t] = answer.comm_out[t-1]:cmul(comm_mask[t-1]:expandAs(answer.comm_out[t-1]))
    	local ask_out = ask_model:forward( {ask.referents, ask.comm_in[t],ask.hid[t-1],ask.cell[t-1]} )
    	----  ask_out = {symbol_prob, act_logprob, baseline, hidstate, cellstate}
        ask.comm_out[t] = ask_out[1]:clone()
        action[t] = sample_multinomial(torch.exp(ask_out[2]))  --(#batch, 1)
        ask.baseline[t] = ask_out[3]:clone():cmul(active[t])
    	ask.hid[t] = ask_out[4]:clone()
        ask.cell[t] = ask_out[5]:clone()

        --comm
        comm_mask[t] = action[t]:eq(2 + g_opts.num_distractors):float():clone()
        --answer
        answer.comm_in[t] = ask.comm_out[t]:cmul(comm_mask[t]:expandAs(ask.comm_out[t]))
        local answer_out = answer_model:forward({answer.target, answer.comm_in[t], answer.hid[t-1],answer.cell[t-1]} )
        ----  answer_out =  symbol_prob, hidstate, cellstate}
        answer.comm_out[t] = answer_out[1]:clone()
        answer.hid[t] = answer_out[2]:clone()
        answer.cell[t] = answer_out[3]:clone()

        batch_act(batch, action[t], active[t])
        reward[t] = batch_reward(batch, active[t])

    end
    local success = batch_success(batch)

    --prepare for GAE
    local delta = {} --TD residual
    delta[g_opts.max_steps] = reward[g_opts.max_steps] - ask.baseline[g_opts.max_steps]
    for t=1, g_opts.max_steps-1 do 
        delta[t] = reward[t] + g_opts.gamma*ask.baseline[t+1] - ask.baseline[t]
    end
    local A_GAE={} --GAE advatage
    A_GAE[g_opts.max_steps] = delta[g_opts.max_steps]
    for t=g_opts.max_steps-1, 1, -1 do 
        A_GAE[t] = delta[t] + g_opts.gamma*g_opts.lambda*A_GAE[t+1] 
    end

    --backward pass
    preproc_paramdx:zero()
    ask_paramdx:zero()
    answer_paramdx:zero()

    ask.grad_hid = torch.Tensor(#batch, g_opts.ask_hidsz):fill(0)
    ask.grad_cell = torch.Tensor(#batch, g_opts.ask_hidsz):fill(0)
    ask.grad_comm_in = torch.Tensor(#batch, g_opts.answer_num_symbols):fill(0)


    answer.grad_hid = torch.Tensor(#batch, g_opts.answer_hidsz):fill(0)
    answer.grad_cell = torch.Tensor(#batch, g_opts.answer_hidsz):fill(0)
    answer.grad_comm_out = torch.Tensor(#batch, g_opts.answer_num_symbols):fill(0)

    local reward_sum = torch.Tensor(#batch):zero() --running reward sum
    for t = g_opts.max_steps, 1, -1 do
        reward_sum:add(reward[t])

        --answer
        local answer_out = answer_model:forward({answer.target, answer.comm_in[t], answer.hid[t-1],answer.cell[t-1]} )
        answer_model:backward({answer.target, answer.comm_in[t], answer.hid[t-1],answer.cell[t-1]}, 
                            {answer.grad_comm_out, answer.grad_hid, answer.grad_cell})
        answer.grad_target = answer_modules['target'].gradInput:clone() --(#batch, inputsz)
        answer.grad_hid = answer_modules['prev_hid'].gradInput:clone()
        answer.grad_cell = answer_modules['prev_cell'].gradInput:clone()
        answer.grad_comm_in = answer_modules['comm_in'].gradInput:clone()

        --comm

        --ask
        ask.grad_comm_out = answer.grad_comm_in:cmul(comm_mask[t]:expandAs(answer.grad_comm_in))

        local R = reward_sum:clone() --(#batch, )
        R:cmul(active[t]) --(#batch, )
        ask.grad_baseline = bl_loss:backward(ask.baseline[t], R):mul(g_opts.alpha):div(#batch)

        ask.grad_action = torch.Tensor(#batch, 2 + g_opts.num_distractors):zero()
        ask.grad_action:scatter(2, action[t], A_GAE[t]:view(-1,1):neg())

        local ask_out = ask_model:forward({ask.referents, ask.comm_in[t],ask.hid[t-1],ask.cell[t-1]})
        ask_model:backward( {ask.referents, ask.comm_in[t],ask.hid[t-1],ask.cell[t-1]},
                            {ask.grad_comm_out, ask.grad_action, ask.grad_baseline, ask.grad_hid, ask.grad_cell})
        ask.grad_referents = ask_modules['referents'].gradInput:clone() --(#batch, 1 + num_distractors, inputsz)
        ask.grad_hid = ask_modules['prev_hid'].gradInput:clone()
        ask.grad_cell = ask_modules['prev_cell'].gradInput:clone()
        ask.grad_comm_in = ask_modules['comm_in'].gradInput:clone()

        answer.grad_comm_out = ask.grad_comm_in:cmul(comm_mask[t-1]:expandAs(ask.grad_comm_in))

        --preporc
        local referents = preproc_model:forward(referents_src_4D)
        preproc_model:backward(referents_src_4D, )


    end

    local stat={}
    stat.reward = reward_sum:sum()
    stat.success = success:sum()
    stat.count = g_opts.batch_size
    return stat

end



function train(N)
    for n = 1, N do
        local stat = {} --for the epoch
		for k = 1, g_opts.nbatches do
			local s = train_batch()
            merge_stat(stat, s)
		end

        for k, v in pairs(stat) do
            if string.sub(k, 1, 5) == 'count' then
                local s = string.sub(k, 6)
                stat['reward' .. s] = stat['reward' .. s] / v
                stat['success' .. s] = stat['success' .. s] / v
            end
        end

        stat.epoch = #g_log + 1
        print(format_stat(stat))
        table.insert(g_log, stat)
    end

end