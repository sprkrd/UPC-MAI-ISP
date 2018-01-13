function J = modify_image(I, attributes)
% modify_image(I, attributes): auxiliary method for data augmentation. It
% receives as a first argument an image (either in grayscale or RGB) and
% applies a series of both color and geometrical transformations to alter
% the image. The modification attributes are specified via the second
% attribute, a struct that may contain the following elements:
%   -   Side: images are cropped to a small Sixe times Side rectangle.
%       Defaults to 256.
%   -   Rotation: rotation of the image (applied before cropping). Defaults
%       to 0 (no rotation.
%   -   ShearX and shearY: shear values
%   -   Gamma: gamma correction. Defaults to 1, or no correction.
%   -   GBlur: std dev of a Gaussian filter applied to the image. Defaults
%       to 0, or no filtering)
%   -   GNoise: variance of the additive Gaussian noise added to the image.
%       Defaults to 0 (no noise).
%   -   SPNoise: percentage of Salt & Pepper noise added to the image.
%       Defaults to 0 (no noise).
    
    %% Defaults
    
    if ~isfield(attributes, 'Side')
        attributes.Side = 256;
    end
    
    if ~isfield(attributes, 'Rotation')
        attributes.Rotation = 0;
    end
    
    if ~isfield(attributes, 'Gamma')
        attributes.Gamma = 1.0;
    end
    
    if ~isfield(attributes, 'GBlur')
        attributes.GBlur = 0.0;
    end
    
    if ~isfield(attributes, 'GNoise')
        attributes.GNoise = 0;
    end
    
    if ~isfield(attributes, 'SPNoise')
        attributes.SPNoise = 0;
    end
    
    if ~isfield(attributes, 'HueAlter')
        attributes.HueAlter = 0;
    end
    
    if ~isfield(attributes, 'shearX')
        attributes.shearX = 0;
    end
    
    if ~isfield(attributes, 'shearY')
        attributes.shearY = 0;
    end
    
    %% Rotate and shear image
    
    theta = attributes.Rotation*pi/180;
    
    xform_rotate = [cos(theta), -sin(theta), 0; sin(theta), cos(theta), 0; 0, 0, 1];
    xform_shear = [1, attributes.shearX, 0; attributes.shearY, 1, 0; 0, 0 ,1];
    tform = affine2d(xform_shear'*xform_rotate');
    
    J = imwarp(I, tform);
    
    % J = imrotate(I, attributes.Rotation, 'bilinear');
    
    %% Crop and resize image
    
    [W, H, ~] = size(J);
    Side = min(W,H);
    if H > W
        slack = floor((H - W)/2);
        J = J(:,slack+1:end-slack,:);
    elseif W > H
        slack = floor((W - H)/2);
        J = J(slack+1:end-slack,:,:);
    end
    J = imresize(J, [attributes.Side, attributes.Side]);
    
    %% Apply gamma correction
    
    if abs(attributes.Gamma - 1.0) > 1e-9
        J = imadjust(J, [], [], attributes.Gamma);
    end
    
    %% Apply Gaussian blur (before noise!)
    
    if attributes.GBlur > 1e-9
        J = imgaussfilt(J, attributes.GBlur);
    end
    
    %% Add Gaussian noise
    
    if attributes.GNoise > 1e-9
        J = imnoise(J, 'gaussian', 0, attributes.GNoise);
    end
    
    %% Alter hue
    
    if abs(attributes.HueAlter) > 1e-9
        J = rgb2hsv(J);
        J(:,:,1) = min(1.0, J(:,:,1)*attributes.HueAlter);
        J = hsv2rgb(J);
    end
    
    %% Add salt & pepper noise
    
    if attributes.SPNoise > 1e-9
        J = imnoise(J, 'salt & pepper', attributes.SPNoise);
    end

end