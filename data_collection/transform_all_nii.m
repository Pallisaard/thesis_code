function transform_all_nii(directory)
    % Check if the directory exists
    if ~isfolder(directory)
        error('Directory does not exist: %s', directory);
    end

    % Get a list of all .nii.gz files in the directory
    nii_files = dir(fullfile(directory, '*.nii.gz'));

    % Loop through each file and process it
    for i = 1:length(nii_files)
        % Get the full path of the file
        file_path = fullfile(directory, nii_files(i).name);

        % Generate the output file name
        [~, name, ext] = fileparts(nii_files(i).name);
        output_name = fullfile(directory, [name '_rotatedf' ext]);

        % Run the nii_tool function with the specified arguments
        nii_xform(file_path, [1, 1, 1], output_name);

        % Remove the original file and rename the output file to the original file name
        delete(file_path);
        movefile(output_name, file_path);
    end
end