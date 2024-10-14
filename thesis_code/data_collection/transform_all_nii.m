function transform_all_nii(directory)
    % Check if the directory exists

    recycle("off");
    if ~isfolder(directory)
        error('Directory does not exist: %s', directory);
    end

    % Get a list of all .nii.gz files in the directory
    nii_files = dir(fullfile(directory, '*.nii.gz'));

    % Loop through each file and process it
    for i = 1:length(nii_files)
        % Get the full path of the file
        file_path = fullfile(directory, nii_files(i).name);
        disp(i);
        disp(file_path);

        % Generate the output file name
        % [~, name, ext] = fileparts(nii_files(i).name);
        % output_name = fullfile(directory, [name ext]);
        % disp(output_name);

        disp("performing transform...");
        % Run the nii_tool function with the specified arguments
        nii_xform(file_path, [1, 1, 1], file_path);

        
        % Remove the original file and rename the output file to the original file name
        % disp("deleting...");
        delete(file_path);
        % disp("moving...");
        % movefile(output_name, file_path);
        disp("");
    end
end