function emdd_split_train_test(cfg)
%EMDD_SPLIT_TRAIN_TEST 按被试 7:3 划分 train_6 / test_6（与 train_test.m 同逻辑）。
arguments
    cfg struct
end
inputFolder = cfg.input_dir;
trainFolder = cfg.train_dir;
testFolder = cfg.test_dir;
trainRatio = cfg.train_ratio;

if ~exist(trainFolder, 'dir'), mkdir(trainFolder); end
if ~exist(testFolder, 'dir'), mkdir(testFolder); end

fileList = dir(fullfile(inputFolder, '*.set'));
subjSet = containers.Map();

for i = 1:length(fileList)
    fname = fileList(i).name;
    base = erase(fname, '.set');
    pos = strfind(base, '_epoch');
    if isempty(pos)
        continue;
    end
    subjID = base(1:pos-1);
    subjSet(subjID) = 1;
end

allSubj = keys(subjSet);
H_subj = {};
MDD_subj = {};
for i = 1:length(allSubj)
    if startsWith(allSubj{i}, 'H_')
        H_subj{end+1} = allSubj{i}; %#ok<AGROW>
    elseif startsWith(allSubj{i}, 'MDD_')
        MDD_subj{end+1} = allSubj{i}; %#ok<AGROW>
    end
end

rng(cfg.random_seed);
H_subj = H_subj(randperm(length(H_subj)));
MDD_subj = MDD_subj(randperm(length(MDD_subj)));

nTrainH = floor(length(H_subj) * trainRatio);
nTrainMDD = floor(length(MDD_subj) * trainRatio);
trainSubj = [H_subj(1:nTrainH), MDD_subj(1:nTrainMDD)];
testSubj = [H_subj(nTrainH+1:end), MDD_subj(nTrainMDD+1:end)];

for i = 1:length(fileList)
    fname = fileList(i).name;
    base = erase(fname, '.set');
    pos = strfind(base, '_epoch');
    if isempty(pos)
        continue;
    end
    subjID = base(1:pos-1);

    src = fullfile(inputFolder, fname);
    [~, nameOnly, ext] = fileparts(fname);
    fdtName = [nameOnly '.fdt'];

    if any(strcmp(trainSubj, subjID))
        copyfile(src, fullfile(trainFolder, fname));
        if exist(fullfile(inputFolder, fdtName), 'file')
            copyfile(fullfile(inputFolder, fdtName), fullfile(trainFolder, fdtName));
        end
    elseif any(strcmp(testSubj, subjID))
        copyfile(src, fullfile(testFolder, fname));
        if exist(fullfile(inputFolder, fdtName), 'file')
            copyfile(fullfile(inputFolder, fdtName), fullfile(testFolder, fdtName));
        end
    end
end

fprintf('[emdd_split] train subjects=%d, test subjects=%d\n', length(trainSubj), length(testSubj));
end
