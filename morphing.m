function morphing

%paths to the input and output images
original_path = "./dataset/Pollen/image/";
fourier_path = "./dataset/Pollen/masks/";
output_path = "./dataset/Pollen/dataset_morphing/";

%parameter for morphing
%alpha is a parameter in range 0-1
%with 2 image a morphing with:
%       alpha 0 create an image equal to image_1
%       alpha 1 create an image equal to image_2
%the algoritm increase alpha every iteration until alpha_min > alpha_max
%for example with alpha_min = alpha_max = 0.5 the output will be an image
%mixed at ~50% 
alpha_min = 0.5;          % must be in range 0~1
alpha_max = 0.5;          % must be in range 0~1
alpha_increment = 0.1;  % must be in range 0.1~0.9
factor_increment = 10;  % factor of increment of original image (es. factor = 10 -> from 350 to 3500), of course if possible
                        % must controll alpha parameters if you create less image than expected

py.morphing_lib.start_morph(original_path, fourier_path, output_path, alpha_min, alpha_max, alpha_increment, factor_increment)

%rename all the image
py.renameDataset.rename_from_morph(output_path)

end