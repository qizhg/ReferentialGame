require 'optim'
require('nn')
require('nngraph')


function train_batch_answer()
    local num_batchs = (epoch_num-1)*g_opts.nbatches + batch_num
	local batch = batch_init(g_opts.batch_size)
    local ask_label, answer_label, ask_comm, answer_comm = batch_comm_label(batch)
	
	--referents
	local ref_input = batch_input(batch)
    --print(ref_input)
    --io.read()
	local preproc_out = preproc_model:forward(ref_input)
    --print(preproc_out)
    --io.read()
    ----  preproc_out = {referents, target}
    ask={}
    ask.grad_referents = preproc_out[1]:clone():zero()

	--forward cache
    active = {}
    action = {}
    reward = {}
    comm_mask = {}
    comm_mask[0] = torch.Tensor(#batch, 1):fill(1)
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

   
    --print(preproc_out[1])
    --print(answer.target)
    --print(batch_target_index(batch):view(-1,1))
    --print(ask_label:view(-1,1))
    --print(answer_label:view(-1,1))
    --print(ask_comm)
    --print(answer_comm)
    --io.read()
    

    

	--forward pass
    --print('---------------batch---------------')
    --print('target index')
    --print(#batch_target_index(batch))
    --print(answer.target)
    --io.read()
    for t = 1, g_opts.max_steps do
        --print('-------t='..t..'----------')
        active[t] = batch_active(batch)
        
        --answer
        answer.comm_in[t] = ask_comm:clone()
        --print(ask_comm)
        --io.read()
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

        --print('reward')
        --print(reward[t]:view(-1,1))

        --io.read()

    end
    local success = batch_success(batch)

    --backward pass
    preproc_paramdx:zero()
    answer_paramdx:zero()

    answer.grad_comm_out = torch.Tensor(#batch, g_opts.answer_num_symbols):fill(0)

    local NLLceriterion = nn.ClassNLLCriterion()

    local avg_err = 0
    for t = g_opts.max_steps, 1, -1 do

        --answer
        ----forward
        local answer_input_table = {}
        answer_input_table[1] = answer.target
        answer_input_table[#answer_input_table+1] = answer.comm_in[t]
        if g_opts.comm == 'Gumbel' then
            answer_input_table[#answer_input_table+1]  = answer.Gumbel_noise[t] 
        end
        local answer_out =  answer_model:forward(answer_input_table):clone()
        --print(answer_out)
        
        local err = NLLceriterion:forward(answer_out,answer_label)
        local grad_p = NLLceriterion:backward(answer_out,answer_label):clone()
        --print(grad_logp)
        --print(answer_out)
        --print(grad_p)
        --io.read()

        avg_err = avg_err + err
    

        answer_model:backward(answer_input_table, grad_p)
        answer.grad_target = answer_modules['target'].gradInput:clone() --(#batch, inputsz)
        answer.grad_comm_in = answer_modules['comm_in'].gradInput:clone()

        --preporc
        preproc_model:forward(ref_input)
        preproc_model:backward(ref_input, {ask.grad_referents,answer.grad_target})

    end

    local stat={}
    stat.avg_err = avg_err/g_opts.nbatches
    return stat

end



function train_answer(N)
    for n = 1, N do
        epoch_num= n
        local x = answer_paramx:clone()
        local stat = {} --for the epoch
		for k = 1, g_opts.nbatches do
            batch_num = k
            xlua.progress(k, g_opts.nbatches)
			local s = train_batch_answer()
            merge_stat(stat, s)
		end

        g_update_param(answer_paramx, answer_paramdx, 'answer')
        g_update_param(preproc_paramx, preproc_paramdx, 'preproc')
        
        local xx = answer_paramx:clone()
        print((x-xx):norm())

        for k, v in pairs(stat) do
            if string.sub(k, 1, 5) == 'count' then
                local s = string.sub(k, 6)
                --stat['reward' .. s] = stat['reward' .. s] / v
                --stat['success' .. s] = stat['success' .. s] / v
                --stat['active' .. s] = stat['active' .. s] / v
                stat['avg_err' .. s] = stat['avg_err' .. s] / v
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