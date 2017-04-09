from graphics import *
from PIL import Image as NewImage


def draw_item(shape, color, size):
	win_sz = 42 
	win = GraphWin('Draw an Item', win_sz, win_sz)
	win.setBackground('white')
	#shape:   0: cross, 1:squar e, 2:triagnle
	#color:   0: red,    1: green, 3: blue
	#size:    0: small,  1:large
	
	if shape == 0:
		if size == 0:
			p1_x = 8
		elif size ==1:
			p1_x = 2
		#else:

		width = win_sz-p1_x-p1_x
		p1 = Point(p1_x,p1_x)
		p2 = Point(p1_x + width,p1_x)
		p3 = Point(p1_x + width,p1_x + width)
		p4 = Point(p1_x, p1_x + width)
		line1 = Line(p1, p3)
		line2 = Line(p2, p4)
		if color == 0:
			line1.setOutline("red")
			line2.setOutline("red")
		elif color == 1:
			line1.setOutline("green")
			line2.setOutline("green")
		elif color == 2:
			line1.setOutline("blue")
			line2.setOutline("blue")
		#else:
		line1.draw(win)
		line2.draw(win)
	
	elif shape == 1:
		if size == 0:
			p1_x = 8
		elif size ==1:
			p1_x = 2
		#else:

		width = win_sz-p1_x-p1_x
		p = Point(p1_x,p1_x)
		pp = Point(p1_x + width,p1_x + width)
		item = Rectangle(p, pp)	
		item.setOutline('white')
		if color == 0:
			item.setFill("red")
		elif color == 1:
			item.setFill("green")
		elif color == 2:
			item.setFill("blue")
		#else:
		item.draw(win)

	elif shape == 2:
		if size == 0:
			p1_x = 8
		elif size ==1:
			p1_x = 2
		#else:

		width = win_sz-p1_x-p1_x
		p1 = Point(p1_x + width/2, p1_x)
		p2 = Point(p1_x + width,p1_x + width)
		p3 = Point(p1_x, p1_x + width)
		item = Polygon([p1, p2, p3])
		item.setOutline('white')
		if color == 0:
			item.setFill("red")
		elif color == 1:
			item.setFill("green")
		elif color == 2:
			item.setFill("blue")
		#else:
		item.draw(win)
	

	filename = ""+ str(shape+1)+str(color+1)+str(size+1)
	win.postscript(file=filename+".eps", colormode='color')
	img = NewImage.open(filename+".eps")
	img.save(filename+".png", "png")

	#win.getMouse()
	win.close()

def main():
	for shape in range(0, 3):
		for color in range(0, 3):
			for size in range(0, 2):
				draw_item(shape, color, size)


#draw_item(0, 2, 0)
main()
