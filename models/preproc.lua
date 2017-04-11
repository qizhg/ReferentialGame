require('nn')
require('nngraph')
paths.dofile('modules/LeNet.lua')

function build_preproc_model()
	local model
	if g_opts.representation == 'image' then
        model = build_preproc_image_model()
    elseif g_opts.representation == 'code' then
        model = build_preproc_code_model()
    end
	return model

end

function build_preproc_image_model()
	local img_src = nn.Identity()() --(#batch, 2+num_distractors, nchannels, height, width)
	local img_src_4D = nn.View(-1,g_opts.nchannels,g_opts.src_height, g_opts.src_width)(img_src)
	local img_embedding = LeNet(img_src_4D)
	local img_embedding_3D =  nn.View(g_opts.batch_size, 2+g_opts.num_distractors, -1)(img_embedding)

	local referents = nn.Narrow(2,1,1+g_opts.num_distractors)(img_embedding_3D)
	local target = nn.Squeeze()(nn.Narrow(2,2+g_opts.num_distractors,1)(img_embedding_3D))

	local model = nn.gModule( {img_src}, {referents, target})
    return model
end

function build_preproc_code_model()
	local code = nn.Identity()() --(#batch, 2 + #distractors, #attr)
	local code_2D = nn.View(-1,g_opts.num_attr)(code)
	local code_2D_embedding = nn.LookupTable(g_opts.attr_range, g_opts.inputsz)(code_2D)
	local code_embedding = nn.View(g_opts.batch_size,2+g_opts.num_distractors,-1)(code_2D_embedding)
	local referents = nn.Narrow(2,1,1+g_opts.num_distractors)(code_embedding)
	local target = nn.Squeeze()(nn.Narrow(2,2+g_opts.num_distractors,1)(code_embedding))

	local model = nn.gModule( {code}, {referents, target})
    return model
end