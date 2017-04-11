require('nn')
require('nngraph')
paths.dofile('modules/LeNet.lua')

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

function build_diff_model()

    
	local referents = nn.Identity()() --(#batch, 1+num_distractors, )
    diff_modules['referents'] = referents.data.module
    
    local referents_flat = nn.View(g_opts.batch_size,-1)(referents)
    local referents_sz = (1 + g_opts.num_distractors) * g_opts.inputsz
	local referents_embedding1 = nonlin()(nn.Linear(referents_sz, g_opts.ask_hidsz)(referents_flat))
    local referents_embedding = nonlin()(nn.Linear(g_opts.ask_hidsz, g_opts.ask_hidsz)(referents_embedding1))


    --local prev_hid = nn.Identity()() --(#batch, ask_hidsz)
    --diff_modules['prev_hid'] = prev_hid.data.module
    --local prev_cell = nn.Identity()() --(#batch, ask_hidsz)
    --diff_modules['prev_cell'] = prev_cell.data.module
    
    --local rnn_input = referents_embedding
    --local hidstate = build_GRU(rnn_input, prev_hid, g_opts.ask_hidsz, g_opts.ask_hidsz)

    local comm_out
    local hid_symbol = nonlin()(nn.Linear(g_opts.ask_hidsz, g_opts.ask_hidsz)(referents_embedding))
    local symbol = nn.Linear(g_opts.ask_hidsz, g_opts.ask_num_symbols)(hid_symbol)
    comm_out = nn.LogSoftMax()(symbol)


    local input_table = {}
    input_table[1] = referents
    --input_table[2] = prev_hid
    --input_table[3] = prev_cell

    local output_table = {}
    output_table[1] = comm_out
    --output_table[2] = hidstate
    --output_table[3] = cellstate
    

    local model = nn.gModule( input_table, output_table)
    return model

end