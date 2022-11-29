function forward_model(data_file, output_dir, cortical_surface_file,...
    mri_file, nas, lpa, rpa, varargin)
% FORWARD_MODEL  Coregistration and forward model computation
%    foward_model(data_file, output_dir, varargin)
%        data_file = file path of data
%        output_dir = directory to put results in
%        cortical_surface_file = file path of cortical surface (gifti)
%        mri_file = file path of MRI (T1)
%        nas = nasion fiducial coil coordinate
%        lpa = left ear fiducial coil coordinate
%        rpa = right ear fiducial coil coordinate
%    Optional arguments (specified by arg name, arg value pairs)
%        eeg_forward_method = forward model algorithm to use for EEG 
%            (default=EEG BEM)
%        meg_forward_method = forward model algorithm to use for MEG
%            (default=Single Shell)
%        patch size = patch size (mm, default=5)
      
defaults = struct('eeg_forward_method', 'EEG BEM',...
    'meg_forward_method', 'Single Shell', 'patch_size', 5);  %define default values
params = struct(varargin{:});
for f = fieldnames(defaults)',
    if ~isfield(params, f{1}),
        params.(f{1}) = defaults.(f{1});
    end
end

spm('defaults','eeg');

spm_jobman('initcfg');

% Create output directory if it does not exist
if exist(output_dir, 'dir')~=7
    mkdir(output_dir);
end

% Create coregistered filename
[filepath,name,ext] = fileparts(data_file);
coreg_file =fullfile(filepath, sprintf('coreg_%s%s', name, ext));

% Smooth mesh
[smoothkern]=spm_eeg_smoothmesh_mm(cortical_surface_file, params.patch_size);

clear jobs
matlabbatch={};
batch_idx=1;

% Copy datafile
matlabbatch{batch_idx}.spm.meeg.other.copy.D = {data_file};
matlabbatch{batch_idx}.spm.meeg.other.copy.outfile = coreg_file;
batch_idx=batch_idx+1;

% Coregister dataset to reconstruction mesh
matlabbatch{batch_idx}.spm.meeg.source.headmodel.D = {coreg_file};
matlabbatch{batch_idx}.spm.meeg.source.headmodel.val = 1;
matlabbatch{batch_idx}.spm.meeg.source.headmodel.comment = '';
matlabbatch{batch_idx}.spm.meeg.source.headmodel.meshing.meshes.custom.mri = {mri_file};
matlabbatch{batch_idx}.spm.meeg.source.headmodel.meshing.meshes.custom.cortex = {cortical_surface_file};
matlabbatch{batch_idx}.spm.meeg.source.headmodel.meshing.meshes.custom.iskull = {''};
matlabbatch{batch_idx}.spm.meeg.source.headmodel.meshing.meshes.custom.oskull = {''};
matlabbatch{batch_idx}.spm.meeg.source.headmodel.meshing.meshes.custom.scalp = {''};
matlabbatch{batch_idx}.spm.meeg.source.headmodel.meshing.meshres = 2;
matlabbatch{batch_idx}.spm.meeg.source.headmodel.coregistration.coregspecify.fiducial(1).fidname = 'nas';
matlabbatch{batch_idx}.spm.meeg.source.headmodel.coregistration.coregspecify.fiducial(1).specification.type = nas;
matlabbatch{batch_idx}.spm.meeg.source.headmodel.coregistration.coregspecify.fiducial(2).fidname = 'lpa';
matlabbatch{batch_idx}.spm.meeg.source.headmodel.coregistration.coregspecify.fiducial(2).specification.type = lpa;
matlabbatch{batch_idx}.spm.meeg.source.headmodel.coregistration.coregspecify.fiducial(3).fidname = 'rpa';
matlabbatch{batch_idx}.spm.meeg.source.headmodel.coregistration.coregspecify.fiducial(3).specification.type = rpa;
matlabbatch{batch_idx}.spm.meeg.source.headmodel.coregistration.coregspecify.useheadshape = 0;
matlabbatch{batch_idx}.spm.meeg.source.headmodel.forward.eeg = params.eeg_forward_method;
matlabbatch{batch_idx}.spm.meeg.source.headmodel.forward.meg = params.meg_forward_method;
spm_jobman('run', matlabbatch);
