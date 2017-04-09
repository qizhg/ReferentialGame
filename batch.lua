function batch_init(size)
    local batch = {}
    for i = 1, size do
        batch[i] = RefGame(g_opts)
    end
    return batch
end

function batch_active(batch)
    local active = torch.Tensor(#batch):zero()
    for i, g in pairs(batch) do
        if g:is_active() then
            active[i] = 1
        end
    end
    return active:view(-1)
end

function batch_input(batch)
    local input = torch.Tensor(#batch, 2+g_opts.num_distractors,g_opts.nchannels,g_opts.src_height, g_opts.src_width)
    input:zero()
    for i, g in pairs(batch) do
        input[i] = g:gen_input()
    end
    return input
end

function batch_reward(batch, active)
    local reward = torch.Tensor(#batch):zero()
    for i, g in pairs(batch) do
        if active[i] == 1 then
            reward[i] = g:get_reward()
        end
    end
    return reward:view(-1)
end