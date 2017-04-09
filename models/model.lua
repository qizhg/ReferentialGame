require('nn')
require('nngraph')
paths.dofile('modules/LeNet.lua')
paths.dofile('answer.lua')
paths.dofile('ask.lua')



function g_init_model()
    preproc_modules = {}
    preproc_model = build_LeNet_model()
    preproc_paramx, preproc_paramdx = preproc_model:getParameters()
    if g_opts.init_std > 0 then
        preproc_paramx:normal(0, g_opts.init_std)
    end

    answer_modules = {}
    answer_model = build_answer_model()
    answer_paramx, answer_paramdx = answer_model:getParameters()
    if g_opts.init_std > 0 then
        answer_paramx:normal(0, g_opts.init_std)
    end

    ask_modules = {}
    ask_model = build_ask_model()
    ask_paramx, ask_paramdx = ask_model:getParameters()
    if g_opts.init_std > 0 then
        ask_paramx:normal(0, g_opts.init_std)
    end
end
