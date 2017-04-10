require('nn')
require('nngraph')
paths.dofile('modules/LeNet.lua')



function build_preproc_model()
	local img_src = nn.Identity()() --(#batch, 2+num_distractors, nchannels, height, width)
	local img_src_4D = nn.View(-1,g_opts.nchannels,g_opts.src_height, g_opts.src_width)(img_src)
	local img_embedding = LeNet(img_src_4D)
	local img_embedding_3D =  nn.View(g_opts.batch_size, 2+g_opts.num_distractors, -1)(img_embedding)

	local referents = nn.Narrow(2,1,1+g_opts.num_distractors)(img_embedding_3D)
	local target = nn.Squeeze()(nn.Narrow(2,2+g_opts.num_distractors,1)(img_embedding_3D))

	local model = nn.gModule( {img_src}, {referents, target})
    return model

end