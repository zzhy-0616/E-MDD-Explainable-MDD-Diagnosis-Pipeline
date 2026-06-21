function emdd_epoch_6s(cfg)
%EMDD_EPOCH_6S 将 ICA 后的 .set 切为 6s epoch（50% 重叠），供 EEGPT step1 使用。
arguments
    cfg struct
end
inputPath = cfg.input_dir;
outputPath = cfg.output_dir;
windowLength = cfg.window_sec;
overlapRatio = cfg.overlap_ratio;

if ~exist(outputPath, 'dir')
    mkdir(outputPath);
end

stepLength = windowLength * (1 - overlapRatio);
fileList = dir(fullfile(inputPath, '*.set'));

for i = 1:length(fileList)
    fileName = fileList(i).name;
    EEG = pop_loadset('filename', fileName, 'filepath', inputPath);
    EEG = eeg_checkset(EEG);

    fs = EEG.srate;
    win_samp = round(windowLength * fs);
    step_samp = round(stepLength * fs);
    total_pnts = EEG.pnts;

    start = 1;
    epoch_num = 1;

    while start + win_samp - 1 <= total_pnts
        epoch_data = EEG.data(:, start : start + win_samp - 1);
        e = EEG;
        e.data = epoch_data;
        e.pnts = win_samp;
        e.times = (0:win_samp-1)/fs * 1000;

        base = strrep(fileName, '.set', '');
        save_name = sprintf('%s_epoch%03d.set', base, epoch_num);
        pop_saveset(e, 'filename', save_name, 'filepath', outputPath);

        start = start + step_samp;
        epoch_num = epoch_num + 1;
    end

    fprintf('[emdd_epoch_6s] %s -> %d epochs\n', fileName, epoch_num - 1);
end
end
