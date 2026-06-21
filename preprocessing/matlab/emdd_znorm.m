function emdd_znorm(cfg)
%EMDD_ZNORM 通道级 Z 归一化：统计量仅来自训练集，避免泄露（与 znorm.m 同逻辑）。
arguments
    cfg struct
end
inputTrain = cfg.input_train;
inputTest = cfg.input_test;
outputTrain = cfg.output_train;
outputTest = cfg.output_test;

if ~exist(outputTrain, 'dir'), mkdir(outputTrain); end
if ~exist(outputTest, 'dir'), mkdir(outputTest); end

trainFiles = emdd_list_h_mdd_sets(inputTrain);
if isempty(trainFiles)
    error('emdd_znorm:EmptyTrain', '训练集为空: %s', inputTrain);
end

EEG = pop_loadset('filename', trainFiles{1});
nChans = EEG.nbchan;
chan_sum = zeros(nChans, 1);
chan_sqsum = zeros(nChans, 1);
total_points = 0;

for f = 1:length(trainFiles)
    EEG = pop_loadset('filename', trainFiles{f});
    data = EEG.data;
    for c = 1:nChans
        sig = squeeze(data(c, :, :));
        sig = sig(:);
        chan_sum(c) = chan_sum(c) + sum(sig);
        chan_sqsum(c) = chan_sqsum(c) + sum(sig.^2);
    end
    total_points = total_points + numel(data(1, :, :));
end

chan_mean = chan_sum / total_points;
chan_std = sqrt(chan_sqsum / total_points - chan_mean.^2);
chan_std(chan_std < 1e-6) = 1;

emdd_normalize_sets(trainFiles, chan_mean, chan_std, outputTrain);
testFiles = emdd_list_h_mdd_sets(inputTest);
emdd_normalize_sets(testFiles, chan_mean, chan_std, outputTest);
fprintf('[emdd_znorm] done: %s , %s\n', outputTrain, outputTest);
end

function files = emdd_list_h_mdd_sets(folder)
dirData = dir(fullfile(folder, '*.set'));
files = {};
for i = 1:length(dirData)
    nm = dirData(i).name;
    if startsWith(nm, 'H_') || startsWith(nm, 'MDD_')
        files{end+1} = nm; %#ok<AGROW>
    end
end
end

function emdd_normalize_sets(fileNames, chan_mean, chan_std, outputFolder)
for f = 1:length(fileNames)
    fname = fileNames{f};
    EEG = pop_loadset('filename', fname);
    data = EEG.data;
    for c = 1:EEG.nbchan
        data(c, :, :) = (data(c, :, :) - chan_mean(c)) / chan_std(c);
    end
    EEG.data = data;
    pop_saveset(EEG, 'filename', fullfile(outputFolder, fname));
end
end
