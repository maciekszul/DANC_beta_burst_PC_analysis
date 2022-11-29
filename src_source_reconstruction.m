function source_reconstruction(data_file, output_dir, d_type, varargin)
% SOURCE_RECONSTRUCTION  Perform source reconstruction
%    source_reconstruction(orig_res4_file, mne_file, epoched)
%        orig_res4_file = file path of the original data 
%        mne_file = file path of the MNE (fif file) to convert
%        epoched = 0 if continuous data, 1 if epoched

defaults = struct('algorithm', 'EBB', 'patch_size', 5, 'n_temp_modes', 4,...
    'woi', [-Inf Inf], 'foi', [0 256]);  %define default values
params = struct(varargin{:});
for f = fieldnames(defaults)'
    if ~isfield(params, f{1})
        params.(f{1}) = defaults.(f{1});
    end
end

spm('defaults','eeg');
spm_jobman('initcfg');

% Setup spatial modes for cross validation
spatialmodesname=fullfile(output_dir, strcat('testmodes_', d_type, '.mat'));
[spatialmodesname,Nmodes,pctest]=spm_eeg_inv_prep_modes_xval(data_file, [], spatialmodesname, 1, 0);

clear jobs
matlabbatch={};
batch_idx=1;

% Source reconstruction
matlabbatch{batch_idx}.spm.meeg.source.invertiter.D = {data_file};
matlabbatch{batch_idx}.spm.meeg.source.invertiter.val = 1;
matlabbatch{batch_idx}.spm.meeg.source.invertiter.whatconditions.all = 1;
matlabbatch{batch_idx}.spm.meeg.source.invertiter.isstandard.custom.invfunc = 'Classic';
matlabbatch{batch_idx}.spm.meeg.source.invertiter.isstandard.custom.invtype = params.algorithm; %;
matlabbatch{batch_idx}.spm.meeg.source.invertiter.isstandard.custom.woi = params.woi;
matlabbatch{batch_idx}.spm.meeg.source.invertiter.isstandard.custom.foi = params.foi;
matlabbatch{batch_idx}.spm.meeg.source.invertiter.isstandard.custom.hanning = 0;
matlabbatch{batch_idx}.spm.meeg.source.invertiter.isstandard.custom.isfixedpatch.randpatch.npatches = 512;
matlabbatch{batch_idx}.spm.meeg.source.invertiter.isstandard.custom.isfixedpatch.randpatch.niter = 1;
matlabbatch{batch_idx}.spm.meeg.source.invertiter.isstandard.custom.patchfwhm = -params.patch_size; %% NB A fiddle here- need to properly quantify
matlabbatch{batch_idx}.spm.meeg.source.invertiter.isstandard.custom.mselect = 0;
matlabbatch{batch_idx}.spm.meeg.source.invertiter.isstandard.custom.nsmodes = Nmodes;
matlabbatch{batch_idx}.spm.meeg.source.invertiter.isstandard.custom.umodes = {spatialmodesname};
matlabbatch{batch_idx}.spm.meeg.source.invertiter.isstandard.custom.ntmodes = params.n_temp_modes;
matlabbatch{batch_idx}.spm.meeg.source.invertiter.isstandard.custom.priors.priorsmask = {''};
matlabbatch{batch_idx}.spm.meeg.source.invertiter.isstandard.custom.priors.space = 1;
matlabbatch{batch_idx}.spm.meeg.source.invertiter.isstandard.custom.restrict.locs = zeros(0, 3);
matlabbatch{batch_idx}.spm.meeg.source.invertiter.isstandard.custom.restrict.radius = 32;
matlabbatch{batch_idx}.spm.meeg.source.invertiter.isstandard.custom.outinv = '';
matlabbatch{batch_idx}.spm.meeg.source.invertiter.modality = {'All'};
matlabbatch{batch_idx}.spm.meeg.source.invertiter.crossval = [pctest 1];
[a,b]=spm_jobman('run', matlabbatch);

% Get results
D=spm_eeg_load(a{1}.D{1});

% Include only good channels
goodchans=D.indchantype('MEG','good');
M=D.inv{1}.inverse.M;
U=D.inv{1}.inverse.U{1};
MU=M*U;
size(MU)
It   = D.inv{1}.inverse.It;
Dgood=squeeze(D(goodchans,It,:));
ntrials=size(Dgood,3);
dlmwrite(fullfile(output_dir, strcat('MU_', d_type,'.tsv')), MU, '\t')
% % Write each trial
% [outpath,outname,outext]=fileparts(a{1}.D{1});
% for t=1:ntrials
%     t
%     trial_data=MU*squeeze(Dgood(:,:,t));
% 
%     c=file_array(fullfile(outpath,sprintf('%s_%d.dat', outname, t)), size(trial_data),'FLOAT32-LE',0,1,0);
%     c(:,:)=trial_data;
% 
%     g = gifti;
%     g.cdata = c;
%     save(g, fullfile(outpath,sprintf('%s_%d.gii', outname, t)), 'ExternalFileBinary');
% end
