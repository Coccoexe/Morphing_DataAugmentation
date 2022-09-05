USAGE GUIDE.
The morphing program itself requires that various conditions be met regarding the arrangement of folders and file names.
For ease of use, I created the scipt in both python and matlab that takes care of creating the folders and renaming the files it needs.
The script needs two parameters:
- path_image -> folder containing the images
- path_masks -> folder containing the respective masks

It is possible to provide the script with images arranged in 3 different options:
Option 1: images already divided into fold and label
Option 2: images divided into fold or label (the script know that there are images under one level of subfolders, it will be up to the user to decide whether to consider them fold or label)
Option 3: all images in one folder

The only constraint is to provide the image-related masks in the same folder as those images.
Masks must be named the same as the respective image.
To better understand this, here is a visual example:

	Option 1:			Option 2:			Option 3:
		path_image			path_image			path_image
			fold_1				fold_1				1.png
		   	   label_1			   1.png			2.png
		   		1.png		   	   2.png			3.png
		   		2.png		   	   ...				...				
		   		...		   	   mask_dir			mask_dir
		      	   	mask_dir			1.png		   	   1.png
		      		   1.png			2.png	 	   	   2.png
		      		   2.png			...			   ...
				   ...			fold_2
		    	   label_2			   1.png
		    		...		   	   2.png
		    	fold_2			   	   ...
		    	   label_1			   mask_dir				
		      		...				1.png			   
		      	   label_2				2.png	
		      		...				...		   
			...				   ...

Using the "data_extraction" script, you don't need to do anything, the images and masks are arranged in the right order so that you can then run "renameDataset"
Once you run the "renameDataset" script, you can easily run the morphing program without difficulty.


MORPHING:

Regarding the morphing program, the operation is as follows:
Taking 2 images A and B, with their respective masks, you create X desired images.
You can decide how many images to create by changing the alpha parameters.
alpha has a range from 0 to 1, 0 the created image is equal to image A and 1 equal to image B.
you can choose how much to increase alpha at each iteration to choose how many images to create.
with alpha_min = alpha_max = 0.5 you create one image by 50% blending the two images.

	A ---- X ---- B
	0     0.5     1  alpha

the available parameters are as follows:

original_path: path of the original images
fourier_path: path of the masks
output_path: output path
alpha_min
alpha_max
alpha_increment
factor_increment = how many images to create compared to the initial ones. (e.g. factor_increment = 10, create 10x initial images)

the method that chooses whether or not to create an image based on factor_increment is a random function with percentage, so you won't necessarily create the exact desired number
possible error should be considered, but the order of magnitude is consistent.

Operation of the choice algorithm:

% corresponding to the probability of choosing an image is = images you want to create / creatable images
the random function will then choose based on this percentage whether to keep the images or not
so for example if I have a dataset with 300 images and I want to create a 10x so 3000 the percentage is 3000 / (combinations of 2 images without repetitions of 300 images) = 3000/44850 = 6.6%
so only 6.6% of the creatable images are actually created. Feeding this data to a random function might then not create exactly 3000 images but a little more or a little less.



