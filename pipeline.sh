#!/bin/bash

while read p; do
  sub_id=$p
  folder='/mnt/ernie_ghassan/datasets/action_modularity/derivatives/fmriprep/'$p
  
  inpf=$folder/$sub_id"_task-action_run-1_desc-preproc_bold.nii.gz"
 
  #printf "calling on Input Dir: $inpdir,\n anatomy: $anat\n with functional data $func\n\n"
  /usr/local/MATLAB/R2023b/bin/matlab -nodisplay -nosplash -nodesktop -r "modularity_pipeline('$inpf', '$sub_id')"
  
done < inp_subj.txt

#/mnt/ernie_ghassan/datasets/action_modularity/derivatives/fmriprep/sub-01/sub-01_ses-mri3t_task-action_run-1_desc-preproc_bold.nii.gz
#/mnt/ernie_ghassan/datasets/action_modularity/derivatives/fmriprep/sub-01/sub-01_task-action_run-1_desc-preproc_bold.nii.gz