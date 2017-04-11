require('nn')
require('nngraph')
paths.dofile('modules/LSTM.lua')

local function nonlin()
    if g_opts.nonlin == 'tanh' then
        return nn.Tanh()
    elseif g_opts.nonlin == 'relu' then
        return nn.ReLU()
    elseif g_opts.nonlin == 'none' then
        return nn.Identity()
    else
        error('wrong nonlin')
    end
end

function build_answer_model()
    local inputsz = g_opts.inputsz
    if g_opts.representation == 'code' then
        inputsz = inputsz * g_opts.num_attr
    end
	local target = nn.Identity()() --(#batch, inputsz)
    answer_modules['target'] = target.data.module
	local target_embedding = nonlin()(nn.Linear(inputsz, g_opts.answer_hidsz)(target))

	local comm_in =  nn.Identity()() --(#batch, ask_num_symbols)
    answer_modules['comm_in'] = comm_in.data.module
	local comm_in_embedding = nonlin()(nn.Linear(g_opts.ask_num_symbols, g_opts.answer_hidsz)(comm_in))

	local embedding = nn.CAddTable()({ target_embedding,  comm_in_embedding })
	

    --local prev_hid = nn.Identity()() --(#batch, answer_hidsz)
    --answer_modules['prev_hid'] = prev_hid.data.module
    --local prev_cell = nn.Identity()() --(#batch, answer_hidsz)
    --answer_modules['prev_cell'] = prev_cell.data.module
	--local hidstate, cellstate = build_lstm(embedding, prev_hid, prev_cell, g_opts.answer_hidsz, g_opts.answer_hidsz)

    local hidstate = embedding

	local comm_out, Gumbel_noise
    local hid_symbol = nonlin()(nn.Linear(g_opts.answer_hidsz, g_opts.answer_hidsz)(hidstate))
    local symbol = nn.Linear(g_opts.answer_hidsz, g_opts.answer_num_symbols)(hid_symbol)
    if g_opts.comm == 'continuous' then 
        comm_out = nn.LogSoftMax()(symbol)
        --comm_out = symbol
    elseif g_opts.comm == 'Gumbel' then
        Gumbel_noise = nn.Identity()()
        local logp = nn.LogSoftMax()(symbol)
        comm_out = build_Gumbel(Gumbel_noise, logp)
    end

    
    local input_table = {}
    input_table[1] = target
    input_table[#input_table+1] = comm_in
    --input_table[#input_table+1] = prev_hid
    --input_table[#input_table+1] = prev_cell
    if g_opts.comm == 'Gumbel' then
        input_table[#input_table+1] = Gumbel_noise
    end

    local output_table = {}
    output_table[1] = comm_out
    --output_table[2] = hidstate
    --output_table[3] = cellstate


    local model = nn.gModule(input_table, output_table)
    return model

end