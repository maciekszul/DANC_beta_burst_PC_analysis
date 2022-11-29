function lagged_coh=run_lagged_coherence(sensor_data, srate, path_file)
% RUN_LAGGED_COHERENCE - Computes lagged coherence across a range of frequencies and lags
%
% Syntax:  run_lagged_coherence(sensor_data, srate)
%
% Inputs:
%    sensor_data - channels x time x trials
%    srate - sampling rate
%
% Outputs:
%    lagged_coh - channels x frequency x cycles
%
% Example: 
%   lagged_coh = compute_lagged_coherence(sensor_data, srate)

% Number of channels
n_chans=size(sensor_data,1);
% Number of time points per trial
n_pts=size(sensor_data,2);
% Number of trials
n_trials=size(sensor_data,3);

% Frequencies to run on. We start at 5Hz because the trials aren't long
% enough to look at lower frequencies over long lags
foi=[5:0.5:100];

% Lags to run on
lags=[1:.05:5];

% Lagged coherence for each subject in each cluster over all frequencies
% and lags
lagged_coh=zeros(n_chans,length(foi),length(lags)).*NaN;

% Create fieldtrip data structure
data=[];
% Channel labels given by cluster names
data.label=cellfun(@num2str,mat2cell([1:n_chans],ones(1,1),ones(n_chans,1)),'uni',false);
% Sampling rate
data.fsample=srate;
data.trial={};
data.time={};
% Trial definition
data.cfg.trl=zeros(n_trials,3);
for t_idx=1:n_trials
    % Trial data
    data.trial{end+1}=sensor_data(:,:,t_idx);
    % Convert trial timestamps to seconds
    data.time{end+1}=linspace(0, n_pts/srate, n_pts);       
    % Create fake trial start time
    data.cfg.trl(t_idx,1)=(t_idx-1)*n_pts+1000;
    % Create fake trial end time
    data.cfg.trl(t_idx,2)=data.cfg.trl(t_idx,1)+n_pts-1;
    % Trial time zero
    data.cfg.trl(t_idx,3)=find(data.time{1,1}==0);
end        
% Empty trialinfo
data.trialinfo=zeros(n_trials,1);
            
% Run lagged coherence
actual_freqs=zeros(length(foi),1);                
l_lagged_coh=zeros(length(data.label),length(foi));

for l_idx=1:length(lags)
    lag=lags(l_idx);                

    parfor f_idx = 1:length(foi)
                                
        % Configuration for frequency analysis
        cfg_F             = [];
        cfg_F.method      = 'mtmconvol';
        cfg_F.taper       = 'hanning';
        cfg_F.output      = 'fourier';
        cfg_F.keeptrials  = 'yes';
        cfg_F.pad         = 'nextpow2';

        % Configuration for lagged coherence
        cfg_LC            = [];            
        cfg_LC.method     = 'laggedcoherence';
        cfg_LC.trialsets  = 'all';
        fs                = srate;            
        cfg_LC.lag        = lag;
        cfg_F.width       = lag;
                    
        % Set freq range
        cfg_F.foi     = foi(f_idx);
        cfg_LC.foi    = foi(f_idx);
            
        % width of time windows for phase comparison (seconds)
        width         = cfg_F.width/cfg_F.foi;                        
        cfg_F.t_ftimwin = width;
                    
        % half width of time window (seconds)
        halfwidth     = ceil(fs*width/2)/fs;

        % Go from half window width after trial start to half window
        % width before trial end
        toi_start     = data.time{1}(1) + halfwidth;
        toi_stop      = data.time{1}(end) - halfwidth;
                    
        % Step size
        step          = ceil(fs*cfg_LC.lag/cfg_F.foi)/fs;
        cfg_F.toi     = toi_start:step:toi_stop;
                    
        % Run frequency analysis
        freqout       = ft_freqanalysis(cfg_F,data);
                    
        % Compute lagged coherence
        lcoh = ft_connectivity_laggedcoherence(cfg_LC,freqout);
                    
        l_lagged_coh(:,f_idx)=lcoh.laggedcoh;
        actual_freqs(f_idx)=lcoh.freq;
    end  
    interped_lc=zeros(size(l_lagged_coh));
    for i=1:size(l_lagged_coh,1)
        interped_lc(i,:)=interp1(actual_freqs,l_lagged_coh(i,:),foi);
    end
    lagged_coh(:,:,l_idx)=interped_lc;
end
save(path_file, 'lagged_coh')

