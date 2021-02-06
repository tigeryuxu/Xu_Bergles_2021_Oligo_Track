(1) Run script with scale_for_animation = 1  (or < 1 for scaling down)

(2) Run imageJ script to convert to combined RGB stacks
(3) load into Vaa3d using "import" >> import general series as timeseries
		- ***use combine across channels

(4) then film animation, in "others"
	- 400 time points per rotation ==> i.e. # of total frames you have
	- 20 secs per rotation

	- 200 frames per rotation


(5) and export as '.BMP'
(6) which can then be converted to .avi by loading into FIJI


***note: will appear in black and white first!!! But when you open up in 3D mode, it will be correct colors








""" FOR CELL TRACKING """
In Vaa3D:
	plugins >> movies >> z_moviemaker

	(1) sampling rate == # of timeframes if you want each cell to be on a frame
	
	(2) choose anchors by moving timeseries as well!!!

	anchors:
	0 degrees, 0 frame

	+ 20 degrees, 50 frames

	0 degrees,    100 frames

	- 20 degrees (340 degrees), 150 frames


	(3) Rotate around X axis???

	(4) set # of z-stacks to remove (0 - 100 kept)


	(5) reset zoom in

	(6) remove bounding boxes, background BCS switch to dark, leave Parallel (scale bar)
		- scale == 0.8302662 xy and 3.0 z


In ImageJ ==> can use >Image >Stacks >Tools >Combine to sync videos side-by-side
	- 10 fps is good

