% Script for generating new data. In order to use it, copy the folder with
% the banknotes images to the directory of this script, and name the copied
% folder 'banknotes'. This folder should contain several subfolders. The
% name of these subfolders is considered to be the target class of the
% images that are inside (e.g. a subfolder called '5' contains images of 5
% euros banknotes). Alternatively, you can create a symbolic link to
% the banknotes folder at this script's location (the symbolic link should
% also be called 'banknotes') or change the value of the 'dataset_folder'
% to the path of the actual banknotes folder (in that case, don't upload
% the changes to the repository, please!).

% The script works as follows: it creates a folder called
% banknotes_augmented in the script's location, and two subfolders called
% train and test. Then it generates 100 modifications of each image in the
% 'banknotes' folder and put 80% of them in the train folder and 20% of
% them in the test folder. All the modifications of a single image are put
% either in 'test' or in 'train', but they are not splitted between them.
% It is possible to alter the number of modifications per image and the
% train/test split by changing the values of the 'modifications_per_image'
% and the 'train_split' variables, respectively. The name convention of the
% saved modifications is 'img_{target}_{imgcount}_{modificationcount}.jpg'.
% Here, 'target' is target class (5, 10, 20 or 50); 'imgcount' is the 
% index of the original image (found under the 'target' subfolder of
% 'banknotes'); and 'modificationcount' is the index of the modification.
% As explained before, all the images for a given 'target' and 'imgcount'
% are either in 'test' or in 'train'.

% In order to learn about the modifications applied to the images, refer to
% the documentation of the 'modify_image.m' function.

rng(42); % For reproducible results

attr = struct('Side', 256, ... % fixed
    'Rotation', 0, ...         % variable
    'Gamma', 1, ...            % variable
    'GBlur', 0.0, ....         % variable
    'GNoise', 0.0, ...,        % variable
    'SPNoise', 0.0, ...        % variable
    'HueAlter', 0.0);          % variable

modifications_per_image = 3;
train_split = 0.6;
validation_split = 0.2;

range_rotation = [0, 360];
range_gamma = [0.5, 2];
range_gblur = [0, 1];
range_gnoise = [-5, -2];
range_spnoise = [-4, -2];
range_huealter = [0.6, 1.4];

dataset_folder = './banknotes';

if ~exist(dataset_folder, 'dir')
    error(['Folder ', dataset_folder, ' does not exist'])
end

output_folder = './banknotes_augmented';
output_folder_train = [output_folder, '/train'];
output_folder_val = [output_folder, '/val'];
output_folder_test = [output_folder, '/test'];

if ~exist(output_folder_train, 'dir')
    mkdir(output_folder_train)
end

if ~exist(output_folder_test, 'dir')
    mkdir(output_folder_test)
end

if ~exist(output_folder_val, 'dir')
    mkdir(output_folder_val)
end

delete([output_folder_train, '/*']);
delete([output_folder_val, '/*']);
delete([output_folder_test, '/*']);


folders = dir(dataset_folder);
start = tic;
for idx = 1:length(folders)
    if folders(idx).name(1) == '.' || ~folders(idx).isdir  % skip ., .., and files
        continue
    end
    target = folders(idx).name;
    target_path = [dataset_folder, '/', target];
    imgcount = 1;
    images = dir(target_path);
    
    N_images = length(images) - 2; % Substract 2 because . and .. folders
    N_images_val = round(validation_split*N_images);
    N_images_train = round(train_split*N_images);
    
    for jdx = 1:length(images)
        if images(jdx).isdir % skip folders
            continue
        end
        image_path = [target_path, '/', images(jdx).name];
        disp(['Now processing ', image_path, '...']);
        elapsed = zeros(1, modifications_per_image);
        I = imread(image_path);
        for modification = 1:modifications_per_image
            tic;
            if modification == 1
                % Just crop the original image (don't blur, rotate, add
                % noise, etc.)
                J = modify_image(I, struct('Side', attr.Side));
            else
                attr.Rotation = range_rotation(1) + rand()*(range_rotation(2) - range_rotation(1));
                attr.Gamma = range_gamma(1) + rand()*(range_gamma(2) - range_gamma(1));
                attr.GBlur = range_gblur(1) + rand()*(range_gblur(2) - range_gblur(1));
                attr.GNoise = 10^(range_gnoise(1) + rand()*(range_gnoise(2) - range_gnoise(1)));
                if attr.GNoise < 1e-4
                    attr.GNoise = 0;
                end
                attr.SPNoise = 10^(range_spnoise(1) + rand()*(range_spnoise(2) - range_spnoise(1)));
                if attr.SPNoise < 1e-4
                    attr.SPNoise = 0;
                end
                attr.HueAlter = range_huealter(1) + rand()*(range_huealter(2) - range_huealter(1));
                J = modify_image(I, attr);
            end
            imgname_out = ['img_', target, '_', num2str(imgcount), '_', num2str(modification), '.jpg'];
            if imgcount <= N_images_train
                output_image_path = [output_folder_train, '/', imgname_out];
            elseif imgcount <= N_images_val + N_images_train
                output_image_path = [output_folder_val, '/', imgname_out];
            else
                output_image_path = [output_folder_test, '/', imgname_out];
            end
            imwrite(J, output_image_path);
            elapsed(modification) = toc;
        end
        disp(['Average time modifying and saving each image: ', num2str(mean(elapsed)), 's']);
        disp(['Elapsed since beginning: ', num2str(toc(start)), 's']);
        imgcount = imgcount + 1;
    end
end
