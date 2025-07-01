%{
Experiment setup for Brain-Computer Interface RSVP.
======================
RSVP Experimental Flow
======================
1. Total of 40 RSVP trials per participant.
2. Each trial:
   a. A stream of 21 characters is presented rapidly at the same screen location.
      - 1 target character (green)
      - 20 non-target characters (white)
      - All shown in randomized order
   b. Presentation rate: 10 Hz (i.e., each character shown for 100 ms)
3. After each stream:
   a. A response window (5 seconds) allows the participant to press a key to indicate the target character.
4. No feedback is given about correctness.
5. Accuracy is measured as T1%:
   - Percentage of correctly identified target characters out of 40 total trials.
=========================
EEG Recording Epoch:
- 200ms pre-stream + 1000ms stream + 1000ms post-stream = 2.2s total.
- This corresponds to 1320 data points per trial at 600 Hz.
=========================
RSVP実験フロー
=========================
1. 被験者ごとに合計40試行のRSVPタスクを実施します。
2. 各試行では以下を行います：
   a. 同じ位置に21文字の文字列を高速で提示します。
      - ターゲット文字が1つ（緑色）
      - 非ターゲット文字が20個（白色）
      - 文字の順序はランダムに並べ替えられます
   b. 示レートは10Hz（各文字の提示時間は100ミリ秒）
3. 文字列の提示終了後：
   a. 5秒間の応答ウィンドウで、参加者がターゲット文字をキーボードで入力します。
4. 正誤のフィードバックは行いません。
5. 正答率（T1%）で成績を評価します：
   - 40試行中、ターゲットを正しく識別できた割合（％）
=========================
EEG記録エポック:
- ストリーム提示前200ミリ秒 + 提示中1000ミリ秒 + 提示後1000ミリ秒 = 合計2.2秒。
- これは600Hzのサンプリングレートで1試行あたり1320サンプルに相当します。
%}
clc; clear; close all;
%% === パス・保存設定 ===
defaultPath = pwd;
subjectName = strtrim(input('Enter Subject Name: ', 's'));
taskName = 'RSVP';
savePath = [defaultPath '\data\' taskName];
mkdir(savePath);
saveFilename = [savePath '\' subjectName '_' taskName '_' datestr(now,30)];
%% === RSVP Parameters ===
all_chars = ['A':'Z'];
n_trials = 30;
stream_len = 10;
stim_duration = 0.1;  % 100 ms
response_window = 5;
%% === EEG設定 ===
filename = "C:\\Users\\ryoii\\OneDrive\\Documents\\GitHub\\P300Speller_2_electrodes\\VIERecorder\\VieOutput\\VieRawData.csv";
opts = detectImportOptions(filename);
opts.SelectedVariableNames = [2 3];
SAMPLE_FREQ_VIE = 600;
FreqWindow = 4:0.5:40;
mtimeWindow = 4;
cplotdata = repmat(0, SAMPLE_FREQ_VIE*mtimeWindow, 3);
cidx = 1;
cbaseline = [0 0 0];
[B1f, A1f] = butter(4, [3/(SAMPLE_FREQ_VIE/2), 40/(SAMPLE_FREQ_VIE/2)]);
Zf1 = [];
timeWindow = 4;
craw = size(readmatrix(filename, opts).*18.3/64, 1);
NoiseMU = [138.5023,138.5023,12.1827,12.1827,103.387,103.387,0.0353,0.0353,10.124,10.124];
NoiseSigma = [123.2406,123.2406,6.0254,6.0254,69.4416,69.4416,0.0039,0.0039,2.899,2.899];
%% === Psychtoolbox画面設定 ===
DisableKeysForKbCheck([]);
[keyIsDown, ~, keyCode] = KbCheck;
if keyIsDown
    DisableKeysForKbCheck(find(keyCode));
end
Screen('Preference', 'SkipSyncTests', 1);
AssertOpenGL;
ScreenDevices = java.awt.GraphicsEnvironment.getLocalGraphicsEnvironment().getScreenDevices();
MainScreen = java.awt.GraphicsEnvironment.getLocalGraphicsEnvironment().getDefaultScreenDevice().getScreen() + 1;
MainBounds = ScreenDevices(MainScreen).getDefaultConfiguration().getBounds();
MonitorPositions = zeros(numel(ScreenDevices), 4);
for n = 1:numel(ScreenDevices)
    Bounds = ScreenDevices(n).getDefaultConfiguration().getBounds();
    MonitorPositions(n,:) = [Bounds.getLocation().getX() + 1, ...
                             -Bounds.getLocation().getY() + 1 - Bounds.getHeight() + MainBounds.getHeight(), ...
                             Bounds.getWidth(), Bounds.getHeight()];
end
windowsize = get(0,'MonitorPositions');
screenid = max(Screen('Screens'));
stimRect = [MonitorPositions(1,1), MonitorPositions(1,2), ...
            MonitorPositions(1,1)+MonitorPositions(1,3), MonitorPositions(1,2)+MonitorPositions(1,4)-250];
[w, rect] = Screen('OpenWindow', screenid, WhiteIndex(screenid)/2, stimRect);
[centerX, centerY] = RectCenter(rect);
HideCursor();
Screen('TextFont', w, 'Courier New');
% --- RESTORED LINE: Set default text size for instructions and UI text ---
Screen('TextSize', w, 25);
Screen('TextStyle', w, 0);
Screen('BlendFunction', w, 'GL_SRC_ALPHA', 'GL_ONE_MINUS_SRC_ALPHA');
Screen('TextFont', w, '-:lang=ja');
% --- ADDED LINE: Disable keyboard input to the command window ---
ListenChar(2);
% EEG plot window
h = figure('Position', [windowsize(1,1), windowsize(1,2), windowsize(1,3), 200], 'Color', 'k');
h.MenuBar = 'none';
h.ToolBar = 'none';
EEGplot = plot([1:SAMPLE_FREQ_VIE*mtimeWindow]/SAMPLE_FREQ_VIE, cplotdata - cbaseline, 'LineWidth', 1);
ha1 = gca;
ha1.GridColor = [1 1 1];
set(gca, 'Color', 'k');
h_yaxis = ha1.YAxis; h_yaxis.Color = 'w'; h_yaxis.Label.Color = [1 1 1];
h_xaxis = ha1.XAxis; h_xaxis.Color = 'w'; h_xaxis.Label.Color = [1 1 1];
xlim([0 4]); ylim([-75 75]);
titletext = title('EEG (0s)', 'Color', 'w', 'FontSize', 22);
xlabel('time (s)'); ylabel('\muV'); yline(0, 'w--');
lgd = legend(EEGplot, {'L', 'R', 'diff'}); lgd.TextColor = [1 1 1];
%% === RSVP Main Loop ===
csvData = {};
Data = {};
cnt = 0;
pre_stream_duration = 0.2;  % 200 ms
post_stream_duration = 1.0; % 1000 ms
stream_duration = stream_len * stim_duration; % 10 characters * 100 ms/char = 1.0s
total_epoch_duration = pre_stream_duration + stream_duration + post_stream_duration; % 0.2 + 1.0 + 1.0 = 2.2s
epoch_samples = round(total_epoch_duration * SAMPLE_FREQ_VIE); % 2.2s * 600Hz = 1320 samples
for T = 1:n_trials
    target_char = all_chars(randi(length(all_chars)));
    nontargets = all_chars(all_chars ~= target_char);
    nontargets = nontargets(randperm(length(nontargets), stream_len - 1));
    stream = [target_char, nontargets];
    stream = stream(randperm(stream_len));
    target_idx = find(stream == target_char);
    % --- NON-TIME-CRITICAL WAIT LOOP (EEG monitoring) ---
    DrawFormattedText(w, double(sprintf('Trial %d / %d\nPress ENTER to start', T, n_trials)), 'center', 'center', WhiteIndex(w));
    Screen('Flip', w);
    baseStart = GetSecs;
    while true
        opts.DataLines = [craw+1 inf];
        tempdataV = readmatrix(filename, opts).*18.3./64;
        if ~isempty(tempdataV)
            tempdataV = [tempdataV tempdataV(:,2)-tempdataV(:,1)];
            craw = craw + size(tempdataV,1);
            [tempdata1,Zf1] = filter(B1f, A1f, tempdataV, Zf1);
            if cidx+size(tempdataV,1)-1 < SAMPLE_FREQ_VIE*timeWindow
                cplotdata(cidx:cidx+size(tempdataV,1)-1,:) = tempdata1;
                cidx = cidx + size(tempdataV,1);
            else
                cplotdata(cidx:end,:) = tempdata1(end-(size(cplotdata,1)-cidx):end,:);
                cidx = 1;
                cbaseline = nanmean(cplotdata);
            end
            [~, cbaseline, cidx] = updateEEGPlotAndNoise(cplotdata, EEGplot, titletext, FreqWindow, SAMPLE_FREQ_VIE, timeWindow, NoiseMU, NoiseSigma, cidx, Zf1, tempdata1, cbaseline);
        end
        [keyIsDown, ~, keyCode] = KbCheck;
        if keyIsDown && keyCode(KbName('Return'))
            break;
        elseif keyIsDown && keyCode(KbName('ESCAPE'))
            CloseAll(w);
            return
        end
    end
    % --- STIMULUS PRESENTATION LOOP (Clean and Time-Critical) ---
    % 3. Wait for the pre-stream duration and record the starting row for data reading.
    WaitSecs(pre_stream_duration);
    start_craw_stream = craw; 
    
    % Initialize array to store precise flip timestamps
    flip_timestamps = zeros(1, stream_len);
    
    for i = 1:stream_len
        char_color = WhiteIndex(w);
        if stream(i) == target_char
            char_color = [0 255 0];
        end
        % --- ADDED LINE: Temporarily set font style to bold for the stimulus ---
        Screen('TextStyle', w, 1); % 1 is for bold
        % --- MODIFIED LINE: Temporarily enlarge the font size for the stimulus character ---
        Screen('TextSize', w, 300);
        DrawFormattedText(w, double(stream(i)), 'center', 'center', char_color);
        % Flip the screen and record the exact time of the flip (in seconds)
        [flip_timestamps(i), ~, ~, ~] = Screen('Flip', w);
        % --- ADDED LINES: Revert font size and style back to the default for subsequent text ---
        Screen('TextSize', w, 25);
        Screen('TextStyle', w, 0); % 0 is for normal
        % Wait for the remaining time to meet the target SOA (0.1s)
        WaitSecs('UntilTime', flip_timestamps(i) + stim_duration);
    end
    
    % --- END OF STIMULUS STREAM ---
    Screen('Flip', w); % Clear the screen
    
    % --- DATA COLLECTION AFTER THE STREAM (Single, Efficient Read) ---
    % 4. Wait for the post-stream duration to ensure all data is written.
    WaitSecs(post_stream_duration); 
    
    opts.DataLines = [start_craw_stream + 1, inf];
    alldataV_temp = readmatrix(filename, opts).*18.3./64;
    
    % Update craw with the total rows read up to this point
    craw = craw + size(alldataV_temp,1);
    % Filter the newly read data
    alldataV = [alldataV_temp, alldataV_temp(:,2) - alldataV_temp(:,1)];
    
    % --- RESPONSIVE WINDOW ---
    DrawFormattedText(w, double('Input target character (5s)...'), 'center', centerY-100, WhiteIndex(w));
    Screen('Flip', w);
    response = '';
    startTime = GetSecs;
    while GetSecs - startTime < response_window
        [keyIsDown, ~, keyCode] = KbCheck;
        if keyIsDown
            key = KbName(find(keyCode));
            if ischar(key) && length(key) == 1 && ismember(upper(key), all_chars)
                response = upper(key);
                break;
            end
        end
    end
    
    % --- SAVE DATA ---
    % 5. Use the new epoch_samples calculation for clipping.
    % Note: The original code saves one clipped data block per trial.
    clipdat = num2cell(alldataV(1:min(end, epoch_samples),:));
    clipdat{1,4} = string(['Trial_' num2str(T) '_Target_' target_char '_Response_' response '_n_' num2str(target_idx)]);
    nSamples = size(clipdat, 1);
    csvOffset = size(csvData, 1);
    csvData(csvOffset + 1 : csvOffset + nSamples, :) = clipdat;
    cnt = cnt + 1;
end
save([saveFilename '_data.mat'], 'Data', 'SAMPLE_FREQ_VIE', 'subjectName');
TESTtable = cell2table(csvData);
TESTtable = renamevars(TESTtable, ["csvData1","csvData2","csvData3","csvData4"], ["LEFT","RIGHT","DIFF","TrialLabel"]);
writetable(TESTtable, [saveFilename '_data.csv']);
CloseAll(w);
function [] = CloseAll(w)
    DrawFormattedText(w, double('実験終了'), 'center', 'center');
    Screen('Flip', w);
    WaitSecs(1);
    close all;
    ShowCursor();
    % --- ALREADY EXISTS: Re-enable keyboard input to the command window ---
    ListenChar(0);
    Screen('CloseAll');
end
function [cNoisez, cbaseline, cidx] = updateEEGPlotAndNoise(cplotdata, EEGplot, titletext, FreqWindow, SAMPLE_FREQ_VIE, timeWindow, NoiseMU, NoiseSigma, cidx, Zf1, tempdata1, cbaseline)
    cdata = cplotdata(:,[1 2]);
    [pxx, ~] = pwelch(cdata, SAMPLE_FREQ_VIE*timeWindow, 0, FreqWindow, SAMPLE_FREQ_VIE);
    cNoiseMat(1:2) = sum(pxx);
    cNoiseMat(3:4) = rms(cdata);
    cNoiseMat(5:6) = max(gradient(cdata));
    cNoiseMat(9:10) = kurtosis(cdata);
    cNoiseMatz = (cNoiseMat - NoiseMU) ./ NoiseSigma;
    cNoisez = [mean(abs(cNoiseMatz([1 3 5 9]))) mean(abs(cNoiseMatz([2 4 6 10])))];
    set(titletext,'String',sprintf('Noise L=%.2f R=%.2f',  cNoisez(1), cNoisez(2)), 'Color','w');
    set(EEGplot(1), 'YData', cplotdata(:,1));
    set(EEGplot(2), 'YData', cplotdata(:,2));
    set(EEGplot(3), 'YData', cplotdata(:,3));
    drawnow;
end