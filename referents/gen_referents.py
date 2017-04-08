from graphics import *
from PIL import Image as NewImage


def draw_item(shape, size, color):
	win = GraphWin('Draw an Item', 32, 32)
	win.setBackground('white')
	#shape:   0: circle, 1: square
	#size:    0: small,  1:large
	#color:   0: red,    1: green 
	
	if shape == 0:
		if size == 0:
			p1_x = 8
		elif size ==1:
			p1_x = 2
		#else:

		width = 32-p1_x-p1_x
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
		#else:
		line1.draw(win)
		line2.draw(win)
	elif shape == 1:
		if size == 0:
			p1_x = 8
		elif size ==1:
			p1_x = 2
		#else:

		width = 32-p1_x-p1_x
		p = Point(p1_x,p1_x)
		pp = Point(p1_x + width,p1_x + width)
		item = Rectangle(p, pp)	
		item.setOutline('white')
		if color == 0:
			item.setFill("red")
		elif color == 1:
			item.setFill("green")
		#else:
		item.draw(win)
	

	filename = ""+ str(shape)+str(size)+str(color)
	win.postscript(file=filename+".eps", colormode='color')
	img = NewImage.open(filename+".eps")
	img.save(filename+".png", "png")

	#win.getMouse()
	win.close()

def main():
	for shape in range(0, 2):
		for size in range(0, 2):
			for color in range(0, 2):
				draw_item(shape, size, color)


main()
