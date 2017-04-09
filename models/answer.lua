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
	local target = nn.Identity()() --(#batch, inputsz)
	local target_embedding = nonlin()(nn.Linear(g_opts.inputsz, g_opts.answer_hidsz)(target))

	local query =  nn.Identity()() --(#batch, ask_num_symbols)
	local query_embedding = nonlin()(nn.Linear(g_opts.ask_num_symbols, g_opts.answer_hidsz)(query))

	local lstm_input = nn.CAddTable()({ target_embedding,  query_embedding })
	local prev_hid = nn.Identity()() --(#batch, answer_hidsz)
    answer_modules['prev_hid'] = prev_hid.data.module
    local prev_cell = nn.Identity()() --(#batch, answer_hidsz)
    answer_modules['prev_cell'] = prev_cell.data.module

	local hidstate, cellstate = build_lstm(lstm_input, prev_hid, prev_cell, g_opts.answer_hidsz, g_opts.answer_hidsz)

	local hid_symbol = nonlin()(nn.Linear(g_opts.answer_hidsz, g_opts.answer_hidsz)(hidstate))
    local symbol = nn.Linear(g_opts.answer_hidsz, g_opts.answer_num_symbols)(hid_symbol)
    local symbol_logprob = nn.LogSoftMax()(symbol)

    local model = nn.gModule({target, query, prev_hid, prev_cell}, 
    						 {symbol_logprob, hidstate, cellstate})
    return model

end