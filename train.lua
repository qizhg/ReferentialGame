require 'optim'
require('nn')
require('nngraph')


function train_batch(task_id)
    local num_batchs = (epoch_num-1)*g_opts.nbatches + batch_num
	local batch = batch_init(g_opts.batch_size)
	
	--referents
	local ref_input = batch_input(batch) 
	local preproc_out = preproc_model:forward(ref_input)
    ----  preproc_out = {referents, target}	

	--forward cache
    active = {}
    action = {}
    reward = {}
    comm_mask = {}
    comm_mask[0] = torch.Tensor(#batch, 1):fill(1)

    ----ask
	local ask = {}
    local ask_hidsz = g_opts.ask_hidsz
    ask.referents = preproc_out[1]:clone()
    ask.comm_in = {}
    ask.comm_out={}
    ask.hid = {} 
    ask.cell = {}
    ask.hid[0] = torch.Tensor(#batch, ask_hidsz):fill(0)
    ask.cell[0] = torch.Tensor(#batch, ask_hidsz):fill(0)
    ask.baseline = {}
    ask.Gumbel_noise = {}

    ----answer
    local answer = {}
    local answer_hidsz = g_opts.answer_hidsz
    answer.target = preproc_out[2]:clone()
    answer.comm_in = {}
    answer.comm_out={}
    answer.comm_out[0] = torch.Tensor(#batch, g_opts.answer_num_symbols):fill(0)
    answer.hid = {} 
    answer.cell = {}
    answer.hid[0] = torch.Tensor(#batch, answer_hidsz):fill(0)
    answer.cell[0] = torch.Tensor(#batch, answer_hidsz):fill(0)
    answer.Gumbel_noise = {}

    

	--forward pass
    --print('---------------batch---------------')
    --print('target index')
    --print(#batch_target_index(batch))
    for t = 1, g_opts.max_steps do
        --print('-------t='..t..'----------')
        active[t] = batch_active(batch)

    	--ask
        ask.comm_in[t] = answer.comm_out[t-1]:cmul(comm_mask[t-1]:expandAs(answer.comm_out[t-1]))
    	local ask_input_table = {}
        ask_input_table[1] = ask.referents
        ask_input_table[#ask_input_table+1] = ask.comm_in[t]
        ask_input_table[#ask_input_table+1] = ask.hid[t-1]
        ask_input_table[#ask_input_table+1] = ask.cell[t-1]
        if g_opts.comm == 'Gumbel' then
            ask.Gumbel_noise[t] = torch.rand(#batch, g_opts.ask_num_symbols):log():neg():log():neg()
            ask_input_table[#ask_input_table+1]  = ask.Gumbel_noise[t] 
        end
        local ask_out = ask_model:forward( ask_input_table )
    	----  ask_out = {comm_out, act_logprob, baseline, hidstate, cellstate}
        ask.comm_out[t] = ask_out[1]:clone()
        action[t] = sample_multinomial(torch.exp(ask_out[2]))  --(#batch, 1)
        --print('action')
        --print(action[t])
        ask.baseline[t] = ask_out[3]:clone():cmul(active[t])
    	ask.hid[t] = ask_out[4]:clone()
        ask.cell[t] = ask_out[5]:clone()

        --comm_mask
        comm_mask[t] = action[t]:eq(2 + g_opts.num_distractors):float():clone()
        
        --answer
        answer.comm_in[t] = ask.comm_out[t]:cmul(comm_mask[t]:expandAs(ask.comm_out[t]))
        local answer_input_table = {}
        answer_input_table[1] = answer.target
        answer_input_table[#answer_input_table+1] = answer.comm_in[t]
        --answer_input_table[#answer_input_table+1] = answer.hid[t-1]
        --answer_input_table[#answer_input_table+1] = answer.cell[t-1]
        if g_opts.comm == 'Gumbel' then
            answer.Gumbel_noise[t] = torch.rand(#batch, g_opts.answer_num_symbols):log():neg():log():neg()
            answer_input_table[#answer_input_table+1]  = answer.Gumbel_noise[t] 
        end
        local answer_out = answer_model:forward(answer_input_table)
        ----  answer_out =  {comm_out, hidstate, cellstate}
        answer.comm_out[t] = answer_out:clone()
        --answer.hid[t] = answer_out[2]:clone()
        --answer.cell[t] = answer_out[3]:clone()

        batch_act(batch, action[t], active[t])
        reward[t] = batch_reward(batch, active[t])
        --print('reward')
        --print(reward[t]:view(-1,1))

        --io.read()

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
    local avg_err = 0
    for t = g_opts.max_steps, 1, -1 do
        reward_sum:add(reward[t])

        --answer
        local answer_input_table = {}
        answer_input_table[1] = answer.target
        answer_input_table[#answer_input_table+1] = answer.comm_in[t]
        --answer_input_table[#answer_input_table+1] = answer.hid[t-1]
        --answer_input_table[#answer_input_table+1] = answer.cell[t-1]
        if g_opts.comm == 'Gumbel' then
            answer_input_table[#answer_input_table+1]  = answer.Gumbel_noise[t] 
        end
        local answer_out =  answer_model:forward(answer_input_table)
        --answer_model:backward(answer_input_table, 
        --                    {answer.grad_comm_out, answer.grad_hid, answer.grad_cell})
        answer_model:backward(answer_input_table, 
                              answer.grad_comm_out)
        answer.grad_target = answer_modules['target'].gradInput:clone() --(#batch, inputsz)
        --answer.grad_hid = answer_modules['prev_hid'].gradInput:clone()
        --answer.grad_cell = answer_modules['prev_cell'].gradInput:clone()
        answer.grad_comm_in = answer_modules['comm_in'].gradInput:clone()

        --comm

        --ask
        ----forward
        local ask_input_table = {}
        ask_input_table[1] = ask.referents
        ask_input_table[#ask_input_table+1] = ask.comm_in[t]
        ask_input_table[#ask_input_table+1] = ask.hid[t-1]
        ask_input_table[#ask_input_table+1] = ask.cell[t-1]
        if g_opts.comm == 'Gumbel' then
            ask_input_table[#ask_input_table+1]  = ask.Gumbel_noise[t] 
        end
        local ask_out = ask_model:forward( ask_input_table )
        ----  ask_out = {comm_out, act_logprob, baseline, hidstate, cellstate}

        ----grad_comm_out
        ask.grad_comm_out = answer.grad_comm_in:cmul(comm_mask[t]:expandAs(answer.grad_comm_in)) --:div(#batch)

        ----grad_bl
        local R = reward_sum:clone() --(#batch, )
        R:cmul(active[t]) --(#batch, )
        ask.grad_baseline = bl_loss:backward(ask.baseline[t], R):mul(g_opts.alpha) --:div(#batch)

        ----grad_action
        ask.grad_action = torch.Tensor(#batch, 2 + g_opts.num_distractors):zero()
        ask.grad_action:scatter(2, action[t], A_GAE[t]:view(-1,1):neg())
        -------entropy
        local beta = g_opts.beta_start - num_batchs*g_opts.beta_start/g_opts.beta_end_batch
        beta = math.max(0,beta)
        local logp = ask_out[2]
        local entropy_grad = logp:clone():add(1)
        entropy_grad:cmul(torch.exp(logp))
        entropy_grad:mul(beta)
        entropy_grad:cmul(active[t]:view(-1,1):expandAs(entropy_grad):clone())
        ask.grad_action:add(entropy_grad) --:div(#batch)
        if g_opts.SL == true then
            ask.grad_baseline:zero()
            ask.grad_action:zero()
            local NLLceriterion = nn.ClassNLLCriterion()
            local action_label = torch.LongTensor(#batch)
            if t<g_opts.max_steps then
                action_label:fill(2+g_opts.num_distractors)
            else
                action_label = batch_target_index(batch):clone()
            end
            local err = NLLceriterion:forward(ask_out[2],action_label)
            avg_err = avg_err + err
            ask.grad_action = NLLceriterion:backward(ask_out[2],action_label)
        end
       

        ask_model:backward( ask_input_table,
                            {ask.grad_comm_out, ask.grad_action, ask.grad_baseline, ask.grad_hid, ask.grad_cell})
        ask.grad_referents = ask_modules['referents'].gradInput:clone() --(#batch, 1 + num_distractors, inputsz)
        ask.grad_hid = ask_modules['prev_hid'].gradInput:clone()
        ask.grad_cell = ask_modules['prev_cell'].gradInput:clone()
        ask.grad_comm_in = ask_modules['comm_in'].gradInput:clone()

        answer.grad_comm_out = ask.grad_comm_in:cmul(comm_mask[t-1]:expandAs(ask.grad_comm_in))

        --preporc
        preproc_model:forward(ref_input)
        preproc_model:backward(ref_input, {ask.grad_referents,answer.grad_target})


    end

    local stat={}
    stat.reward = reward_sum:sum()
    stat.success = success:sum()
    stat.count = g_opts.batch_size
    stat.avg_err = avg_err
    --stat.active = active[2]:sum()
    return stat

end



function train(N)
    for n = 1, N do
        epoch_num= n
        local x = ask_paramx:clone()
        local stat = {} --for the epoch
		for k = 1, g_opts.nbatches do
            batch_num = k
            xlua.progress(k, g_opts.nbatches)
			local s = train_batch()
            merge_stat(stat, s)
		end

        g_update_param(preproc_paramx, preproc_paramdx, 'preproc')
        g_update_param(answer_paramx, answer_paramdx, 'answer')
        g_update_param(ask_paramx, ask_paramdx, 'ask' )

        local xx = ask_paramx:clone()
        print((x-xx):norm())

        for k, v in pairs(stat) do
            if string.sub(k, 1, 5) == 'count' then
                local s = string.sub(k, 6)
                stat['reward' .. s] = stat['reward' .. s] / v
                stat['success' .. s] = stat['success' .. s] / v
                --stat['active' .. s] = stat['active' .. s] / v
                --stat['avg_err' .. s] = stat['avg_err' .. s] / v
            end
        end

        stat.epoch = #g_log + 1
        print(format_stat(stat))
        table.insert(g_log, stat)
    end

end

function g_update_param(x, dx, model_name)
    local f = function(x0) return x, dx end
    if not g_optim_state then
        g_optim_state = {}
        for i = 1, #model_id2name do
            g_optim_state[i] = {} 
        end
    end
    local model_id = model_name2id[model_name]
    local config = {learningRate = g_opts.lrate}
    if g_opts.optim == 'sgd' then
        config.momentum = g_opts.momentum
        config.weightDecay = g_opts.wdecay
        optim.sgd(f, x, config, g_optim_state[model_id])
    elseif g_opts.optim == 'rmsprop' then
        config.alpha = g_opts.rmsprop_alpha
        config.epsilon = g_opts.rmsprob_eps
        config.weightDecay = g_opts.wdecay
        optim.rmsprop(f, x, config, g_optim_state[model_id])
    elseif g_opts.optim == 'adam' then
        config.beta1 = g_opts.adam_beta1
        config.beta2 = g_opts.adam_beta2
        config.epsilon = g_opts.adam_eps
        optim.adam(f, x, config, g_optim_state[model_id])
    else
        error('wrong optim')
    end

end