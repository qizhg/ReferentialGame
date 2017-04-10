local RefGame = torch.class('RefGame')


function RefGame:__init(opts)
	self.num_distractors = opts.num_distractors


	--attributes
	self.attr = {}
	table.insert(self.attr, 'shape')
	table.insert(self.attr, 'color')
	table.insert(self.attr, 'size')

	self.shape_range = opts.shape_range
	self.color_range = opts.color_range
	self.size_range = opts.size_range

	self.referents={}

	self.referents[1]={}
	self.referents[1].shape = torch.random(1, self.shape_range)
	self.referents[1].color = torch.random(1, self.color_range)
	self.referents[1].size = torch.random(1, self.size_range)

	self.referents[2]={}
	for k,v in pairs(self.referents[1]) do
    	self.referents[2][k] = v
    end
	local distracting_attr = self.attr[torch.random(1, #self.attr)]
	local distracting
	for i = 1, 100 do
		distracting = torch.random(1, self[distracting_attr..'_range'])
		if distracting ~= self.referents[1][distracting_attr] then break end
    end
    if distracting == self.referents[1][distracting_attr] then
    	error('failed 100 times to find attr for distractor')
    end
    self.referents[2][distracting_attr] = distracting
    self.target_index = torch.random(1, 1 + self.num_distractors)


    --finish & success
    self.finished = false
    self.success = 0 --0: not finised, 1: success, -1 unsecueess

    --cost
    self.costs = {}
    self.costs.query = opts.cost_query
    self.costs.correct = opts.cost_correct
    self.costs.wrong = opts.cost_wrong
end

function RefGame:gen_input_image()
	--[....target...; target]
	local input = torch.Tensor(2+self.num_distractors,g_opts.nchannels,g_opts.src_height, g_opts.src_width)
	for i = 1, 1+self.num_distractors do
		local ref_name = ''..self.referents[i].shape..self.referents[i].color..self.referents[i].size
		input[i] = referents_src[ref_name]:clone()
	end
	local target_name = ''..self.referents[self.target_index].shape..self.referents[self.target_index].color..self.referents[self.target_index].size
	input[2+self.num_distractors] = referents_src[target_name]:clone()
	return input
end

function RefGame:gen_input_code()
	--[....target...; target]
	local input = torch.Tensor(2+self.num_distractors, #self.attr)
	for i = 1, 1+self.num_distractors do
		for attr_id = 1, #self.attr do 
			input[i][attr_id] = self.referents[i][self.attr[attr_id]]
		end
	end

	for attr_id = 1, #self.attr do 
		input[2+self.num_distractors][attr_id] = self.referents[self.target_index][self.attr[attr_id]]
	end
	return input
end

function RefGame:act(action)
	if self.finished == true then 
		return 
	end

	if action >= 1 and action <= 1 + self.num_distractors then
		self.finished = true
		if action == self.target_index then 
			self.success = 1
		else
			self.success = -1
		end
	else
		--query 
	end
end


function RefGame:get_reward()
    if self.success == 1 then
        return -self.costs.correct
    elseif self.success == -1 then
        return -self.costs.wrong
    else --self.success == 0
    	return -self.costs.query
    end
end

function RefGame:is_finish()
    if self.finished == true then
        return true
    else
        return false
    end
end

function RefGame:is_active()
    if self.finished == true then
        return false
    else
        return true
    end
end

function RefGame:is_success()
    if self.success == 1 then
        return true
    else
        return false
    end
end
