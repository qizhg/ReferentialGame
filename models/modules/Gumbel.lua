require('nn')
require('nngraph')

function build_Gumbel_SoftMax_model()
    local Gumbel_noise = nn.Identity()()
    local logp =  nn.Identity()()
    local Gumbel_trick = nn.CAddTable()({Gumbel_noise, logp})
    local Gumbel_trick_temp = nn.MulConstant(1.0/g_opts.Gumbel_temp)(Gumbel_trick)
    local Gumbel_SoftMax = nn.SoftMax()(Gumbel_trick_temp)
    local model = nn.gModule({logp, Gumbel_noise}, {Gumbel_SoftMax})
    return model
end