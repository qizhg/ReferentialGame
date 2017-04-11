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
    local input
    if g_opts.representation == 'image' then
        input = batch_input_image(batch)
    elseif g_opts.representation == 'code' then
        input = batch_input_code(batch)
    end
    return input

end
function batch_input_image(batch)
    local input = torch.Tensor(#batch, 2+g_opts.num_distractors,g_opts.nchannels,g_opts.src_height, g_opts.src_width)
    input:zero()
    for i, g in pairs(batch) do
        input[i] = g:gen_input_image()
    end
    return input
end
function batch_input_code(batch)
    local input = torch.Tensor(#batch, 2+g_opts.num_distractors, g_opts.num_attr)
    input:zero()
    for i, g in pairs(batch) do
        input[i] = g:gen_input_code()
    end
    return input
end

function batch_act(batch, action, active)
    for i, g in pairs(batch) do
        if active[i] == 1 then
            g:act(action[i][1])
        end
    end
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

function batch_success(batch)
    local success = torch.Tensor(#batch):fill(0)
    for i, g in pairs(batch) do
        if g:is_success() then
            success[i] = 1
        end
    end
    return success
end

function batch_target_index(batch)
    local target_index = torch.Tensor(#batch):fill(0)
    for i, g in pairs(batch) do
        target_index[i] = g.target_index
    end
    return target_index
end

function batch_comm_label(batch)
    local ask_label = torch.Tensor(#batch):fill(0)
    local answer_label = torch.Tensor(#batch):fill(0)
    local ask_comm = torch.Tensor(#batch, g_opts.shape_range):fill(0)
    local answer_comm = torch.Tensor(#batch, g_opts.shape_range):fill(0)
    for i, g in pairs(batch) do
        for attr_id, attr in pairs(g.attr) do
            if g.referents[1][attr] ~= g.referents[2][attr] then
                ask_label[i] = attr_id
                answer_label[i] = g.referents[g.target_index][attr]
                ask_comm[i][attr_id] = 1
                answer_comm[i][g.referents[g.target_index][attr]] = 1 
            end
        end
    end
    return ask_label, answer_label, ask_comm, answer_comm
end