run_fastsurfer() {
    # Check if the number of arguments is exactly 2
    if [ "$#" -ne 2 ]; then
        echo "Error: Exactly two arguments are required."
        echo "Usage: run_fastsurfer <input_file> <subjectId>"
        return 1
    fi

    # if $1 ends in .nii.gz, then use $1 as the input file
    # else use $1.nii.gz as the input file
    if [[ $1 == *.nii.gz ]]; then
        local input_file=$1
    else
        local input_file=$1.nii.gz
    fi

    local subjectId=$2

    singularity exec --nv \
                    --no-home \
                    -B /home/projects/thesis/data:/data \
                    -B /home/projects/thesis/fastsurfer_output:/output \
                    -B /.fastsurfer:/fs_license \
                    ./fastsurfer-gpu.sif \
                    /fastsurfer/run_fastsurfer.sh \
                    --fs_license /fs_license/license.txt \
                    --t1 /data/${input_file} \
                    --sid ${subjectId} --sd /output \
                    --py python3 --seg_only
}