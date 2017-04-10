require('nn')
require('nngraph')
paths.dofile('preproc.lua')
paths.dofile('answer.lua')
paths.dofile('ask.lua')



function g_init_model()
    model_id2name = {}
    model_name2id = {}
    model_id2name[1] = 'preproc'
    model_id2name[2] = 'ask'
    model_id2name[3] = 'answer'
    model_name2id['preproc'] = 1
    model_name2id['ask'] = 2
    model_name2id['answer'] = 3

    preproc_modules = {}
    preproc_model = build_preproc_model()
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
    bl_loss = nn.MSECriterion()
end
