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

function build_ask_model()

	local referents = nn.Identity()() --(#batch, 1 + num_distractors, inputsz)
    ask_modules['referents'] = referents.data.module
	local referents_sz = (1 + g_opts.num_distractors) * g_opts.inputsz
	local referents_flat = nn.View(referents_sz):setNumInputDims(2)(referents)
	local referents_embedding = nonlin()(nn.Linear(referents_sz, g_opts.ask_hidsz)(referents_flat))

	local comm_in =  nn.Identity()() --(#batch, answer_num_symbols)
    ask_modules['comm_in'] = comm_in.data.module
	local answer_embedding = nonlin()(nn.Linear(g_opts.answer_num_symbols, g_opts.ask_hidsz)(comm_in))

	local lstm_input = nn.CAddTable()({ referents_embedding,  answer_embedding })
	local prev_hid = nn.Identity()() --(#batch, ask_hidsz)
    ask_modules['prev_hid'] = prev_hid.data.module
    local prev_cell = nn.Identity()() --(#batch, ask_hidsz)
    ask_modules['prev_cell'] = prev_cell.data.module

	local hidstate, cellstate = build_lstm(lstm_input, prev_hid, prev_cell, g_opts.ask_hidsz, g_opts.ask_hidsz)

	local hid_symbol = nonlin()(nn.Linear(g_opts.ask_hidsz, g_opts.ask_hidsz)(hidstate))
    local symbol = nn.Linear(g_opts.ask_hidsz, g_opts.ask_num_symbols)(hid_symbol)
    local symbol_prob = nn.SoftMax()(symbol)

    local hid_act = nonlin()(nn.Linear(g_opts.ask_hidsz, g_opts.ask_hidsz)(hidstate))
    local act = nn.Linear(g_opts.ask_hidsz, 1 + 1 + g_opts.num_distractors)(hid_act)
    local act_logprob = nn.LogSoftMax()(act)

    local hid_bl = nonlin()(nn.Linear(g_opts.ask_hidsz, g_opts.ask_hidsz)(hidstate))
    local baseline = nn.Linear(g_opts.ask_hidsz, 1)(hid_bl)



    local model = nn.gModule( {referents, comm_in, prev_hid, prev_cell},
    						 {symbol_prob, act_logprob, baseline, hidstate, cellstate})
    return model

end