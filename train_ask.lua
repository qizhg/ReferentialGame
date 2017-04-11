require 'optim'
require('nn')
require('nngraph')


function train_batch_ask()
    local num_batchs = (epoch_num-1)*g_opts.nbatches + batch_num
	local batch = batch_init(g_opts.batch_size)
    local ask_label, answer_label, ask_comm, answer_comm = batch_comm_label(batch)
	
	--referents
	local ref_input = batch_input(batch) 
	local preproc_out = preproc_model:forward(ref_input)
    ----  preproc_out = {referents, target}
    --print(batch_input_code(batch))
    --io.read()

    answer={}
    answer.grad_target = preproc_out[2]:clone():zero()

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


    ask.comm_in[1] = answer_comm:clone():zero()
    ask.comm_in[2] = answer_comm:clone():zero()


	--forward pass
    --print('---------------batch---------------')
    --print('target index')
    --print(#batch_target_index(batch))
    for t = 1, g_opts.max_steps do
        --print('-------t='..t..'----------')
        active[t] = batch_active(batch)

    	--ask
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

        --batch_act(batch, action[t], active[t])
        reward[t] = batch_reward(batch, active[t])
        --print('reward')
        --print(reward[t]:view(-1,1))

        --io.read()

    end
    local success = batch_success(batch)

    --backward pass
    preproc_paramdx:zero()
    ask_paramdx:zero()

    ask.grad_hid = torch.Tensor(#batch, g_opts.ask_hidsz):fill(0)
    ask.grad_cell = torch.Tensor(#batch, g_opts.ask_hidsz):fill(0)
    ask.grad_comm_in = torch.Tensor(#batch, g_opts.answer_num_symbols):fill(0)

    local reward_sum = torch.Tensor(#batch):zero() --running reward sum
    local avg_action_err = 0
    local avg_comm_err = 0
    local NLLceriterion = nn.ClassNLLCriterion()

    for t = g_opts.max_steps, 1, -1 do
        reward_sum:add(reward[t])

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
        local grad_comm_out = ask_out[1]:clone():zero()
        if t==1 then
            local comm_err = NLLceriterion:forward(ask_out[1],ask_label)
            grad_comm_out = NLLceriterion:backward(ask_out[1],ask_label):clone()
            avg_comm_err = avg_comm_err + comm_err
        elseif t==2 then
            local comm_err = NLLceriterion:forward(ask_out[1],ask_label)
            grad_comm_out = NLLceriterion:backward(ask_out[1],ask_label):clone()
            avg_comm_err = avg_comm_err + comm_err
        end

        ----grad_bl
        local R = reward_sum:clone() --(#batch, )
        R:cmul(active[t]) --(#batch, )
        ask.grad_baseline = bl_loss:backward(ask.baseline[t], R):mul(g_opts.alpha) --:div(#batch)

        --grad_action_logp
        local grad_action_logp = ask_out[2]:clone():zero()
        local action_label = torch.LongTensor(#batch)
        if t==1 then
            action_label:fill(2+g_opts.num_distractors)
        else
            action_label = batch_target_index(batch):clone()
        end
        
        local action_err = NLLceriterion:forward(ask_out[2],action_label)
        grad_action_logp = NLLceriterion:backward(ask_out[2],action_label):clone()
        avg_action_err = avg_action_err + action_err
        
       
        grad_action_logp:zero()
        ask.grad_baseline:zero()
        grad_comm_out:div(#batch)
        ask_model:backward( ask_input_table,
                            {grad_comm_out, grad_action_logp, ask.grad_baseline, ask.grad_hid, ask.grad_cell})
        ask.grad_referents = ask_modules['referents'].gradInput:clone() --(#batch, 1 + num_distractors, inputsz)
        ask.grad_hid = ask_modules['prev_hid'].gradInput:clone()
        ask.grad_cell = ask_modules['prev_cell'].gradInput:clone()
        ask.grad_comm_in = ask_modules['comm_in'].gradInput:clone()

        --preporc
        preproc_model:forward(ref_input)
        preproc_model:backward(ref_input, {ask.grad_referents,answer.grad_target})

    end

    local stat={}
    stat.reward = reward_sum:sum()
    stat.success = success:sum()
    stat.count = g_opts.batch_size
    stat.avg_action_err = avg_action_err
    stat.avg_comm_err = avg_comm_err
    --stat.active = active[2]:sum()
    return stat

end



function train_ask(N)
    for n = 1, N do
        epoch_num= n
        local x = ask_paramx:clone()
        local stat = {} --for the epoch
		for k = 1, g_opts.nbatches do
            batch_num = k
            xlua.progress(k, g_opts.nbatches)
			local s = train_batch_ask()
            merge_stat(stat, s)
		end

        g_update_param(preproc_paramx, preproc_paramdx, 'preproc')
        g_update_param(ask_paramx, ask_paramdx, 'ask' )

        local xx = ask_paramx:clone()
        --print((x-xx):norm())

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