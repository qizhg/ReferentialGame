require 'optim'
require('nn')
require('nngraph')


function train_batch_diff()
    local num_batchs = (epoch_num-1)*g_opts.nbatches + batch_num
	local batch = batch_init(g_opts.batch_size)
    local ask_label, answer_label, ask_comm, answer_comm = batch_comm_label(batch)
	
	

    --referents
    local ref_input = batch_input(batch) 

    local hid = {} 
    local cell = {}
    hid[0] = torch.Tensor(#batch, g_opts.ask_hidsz):fill(0)
    cell[0] = torch.Tensor(#batch, g_opts.ask_hidsz):fill(0.01)
    --forward
    local preproc_out = preproc_model:forward(ref_input)
    ----  preproc_out = {referents, target}
    local grad_target = preproc_out[2]:clone():zero()
    

    local input_table = {}
    input_table[1] = preproc_out[1]
    input_table[#input_table+1] = hid[0]
    --input_table[#input_table+1] = cell[0]
    local diff_out = diff_model:forward(input_table)
    local test = sample_multinomial(torch.exp(diff_out[1]))
    local ct = test:squeeze():eq(ask_label:long())

    --backward pass
    diff_paramdx:zero()
    preproc_paramdx:zero()

    grad_hid = torch.Tensor(#batch, g_opts.ask_hidsz):fill(0)
    grad_cell = torch.Tensor(#batch, g_opts.ask_hidsz):fill(0)
    local NLLceriterion = nn.ClassNLLCriterion()
    local err = NLLceriterion:forward(diff_out[1],ask_label)
    local grad_comm = NLLceriterion:backward(diff_out[1],ask_label):clone()
    --diff_model:backward(preproc_out[1],grad_comm )
    diff_model:backward(input_table,{grad_comm,grad_hid} )
    --diff_model:backward(input_table,{grad_comm,grad_hid,grad_cell} )
    
    grad_referents = diff_modules['referents'].gradInput:clone()
    preproc_model:backward(ref_input, {grad_referents, grad_target})

    


    local stat={}
    stat.avg_err = err
    stat.pred = ct:sum()/g_opts.batch_size
    return stat

end



function train_diff(N)
    for n = 1, N do
        epoch_num= n
        local x = diff_paramx:clone()
        local stat = {} --for the epoch
		for k = 1, g_opts.nbatches do
            batch_num = k
            xlua.progress(k, g_opts.nbatches)
			local s = train_batch_diff()
            merge_stat(stat, s)
		end

        g_update_param(diff_paramx, diff_paramdx, 'diff')
        g_update_param(preproc_paramx, preproc_paramdx, 'preproc')
        
        local xx = diff_paramx:clone()
        print((x-xx):norm())

        for k, v in pairs(stat) do
            if string.sub(k, 1, 5) == 'count' then
                local s = string.sub(k, 6)
                --stat['reward' .. s] = stat['reward' .. s] / v
                --stat['success' .. s] = stat['success' .. s] / v
                --stat['active' .. s] = stat['active' .. s] / v
                --stat['avg_err' .. s] = stat['avg_err' .. s] / v
                
            end
        end
        stat['pred'] = stat['pred'] / g_opts.nbatches

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