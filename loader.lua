require('image')

referents_src = {}

for shape = 1, 3 do 
	for color  = 1, 3 do 
		for size = 1, 2 do
			local ref_name = ''..shape..color..size
			referents_src[ref_name] = image.load('referents/'..ref_name..'.png',3,'byte')
		end
	end
end