% This script runs a P300 speller experiment by flashing rows and columns,
% recording EEG data, calling a Python model for prediction, and displaying the result.

% --- Cleanup and Assertions ---
sca;
close all;
clearvars;

% Unify key names across different operating systems
KbName('UnifyKeyNames');
escapeKey = KbName('ESCAPE');
spaceKey = KbName('space');

try
    % --- Words to Spell ---
    wordsToSpell = {'KEIO', 'SFC', '2025S'};

    % --- Experiment Parameters ---
    flashDuration = 0.1; % 100 ms
    isiDuration = 0.05;  % 50 ms Inter-Stimulus Interval
    totalFlashes = 150;
    numRows = 6;
    numCols = 6;
    numSequences = numRows + numCols; % 12 total sequences
    
    preStreamMs = 200;
    postStreamMs = 1000;
    streamDurationSecs = totalFlashes * flashDuration;
    totalEpochSecs = (preStreamMs / 1000) + streamDurationSecs + (postStreamMs / 1000);

    % --- EEG Parameters ---
    eegCsvPath = "C:\Users\ryoii\OneDrive\Documents\GitHub\P300Speller_2_electrodes\VIERecorder\VieOutput\VieRawData.csv";
    opts = detectImportOptions(eegCsvPath);
    opts.SelectedVariableNames = [2 3]; % Select the two EEG channels
    SAMPLE_FREQ = 600;
    EXPECTED_SAMPLES = round(totalEpochSecs * SAMPLE_FREQ); % Should be 9720
    
    % Plotting window parameters
    plotTimeWindow = 4;
    plotDataBuffer = repmat(0, SAMPLE_FREQ * plotTimeWindow, 3);
    plotIndex = 1;
    plotBaseline = [0 0 0];
    
    % Filter setup
    [Bf, Af] = butter(4, [3/(SAMPLE_FREQ/2), 40/(SAMPLE_FREQ/2)]);
    Zf = [];
    
    % Initialize the row counter for the CSV file
    try
        initialData = readmatrix(eegCsvPath, opts);
        lastReadRow = size(initialData, 1);
    catch
        lastReadRow = 0;
    end
    
    % Noise parameters
    NoiseMU = [138.5023,138.5023,12.1827,12.1827,103.387,103.387,0.0353,0.0353,10.124,10.124];
    NoiseSigma = [123.2406,123.2406,6.0254,6.0254,69.4416,69.4416,0.0039,0.0039,2.899,2.899];

    % --- Screen Setup ---
    Screen('Preference', 'SkipSyncTests', 1);
    screens = Screen('Screens');
    screenNumber = max(screens);

    % Define colors
    backgroundColor = [0 0 0];
    textColor = [128 128 128];
    highlightColor = [255 255 255];

    % Open a window
    [window, ~] = Screen('OpenWindow', screenNumber, backgroundColor);
    [screenXpixels, screenYpixels] = Screen('WindowSize', window);
    ListenChar(2);
    HideCursor;

    % Set text properties
    Screen('TextFont', window, 'Courier New');
    Screen('TextStyle', window, 1);

    % --- EEG Plot Window Setup ---
    windowsize = get(0,'MonitorPositions');
    h = figure('Position', [windowsize(1,1), windowsize(1,2), windowsize(1,3), 200], 'Color', 'k');
    h.MenuBar = 'none'; h.ToolBar = 'none';
    EEGplot = plot( (1:SAMPLE_FREQ*plotTimeWindow)/SAMPLE_FREQ, plotDataBuffer - plotBaseline, 'LineWidth', 1);
    ha1 = gca; ha1.GridColor = [1 1 1]; set(gca, 'Color', 'k');
    h_yaxis = ha1.YAxis; h_yaxis.Color = 'w'; h_yaxis.Label.Color = [1 1 1];
    h_xaxis = ha1.XAxis; h_xaxis.Color = 'w'; h_xaxis.Label.Color = [1 1 1];
    xlim([0 plotTimeWindow]); ylim([-75 75]);
    titletext = title('EEG (0s)', 'Color', 'w', 'FontSize', 22);
    xlabel('time (s)'); ylabel('\muV'); yline(0, 'w--');
    lgd = legend(EEGplot, {'L', 'R', 'diff'}); lgd.TextColor = [1 1 1];

    % --- Speller Matrix Definition ---
    spellerMatrix = {'A','B','C','D','E','F'; 'G','H','I','J','K','L'; 'M','N','O','P','Q','R'; 'S','T','U','V','W','X'; 'Y','Z','0','1','2','3'; '4','5','6','7','8','9'};
    [numRows, numCols] = size(spellerMatrix);

    % --- Grid Layout Calculation ---
    gridWidth = screenXpixels * 0.9;
    gridHeight = screenYpixels * 0.9;
    gridX_start = (screenXpixels - gridWidth) / 2;
    gridY_start = (screenYpixels - gridHeight) / 2;
    cellWidth = gridWidth / numCols;
    cellHeight = gridHeight / numRows;

    % --- Main Experiment Loop (Iterate through words) ---
    for wordIdx = 1:length(wordsToSpell)
        currentTargetWord = wordsToSpell{wordIdx};
        predictedWord = '';
        
        fprintf('\n==================================\n');
        fprintf('Target Word: %s\n', currentTargetWord);
        fprintf('==================================\n');

        % --- Letter Loop (Iterate through characters of the current word) ---
        for letterIdx = 1:length(currentTargetWord)
            
            % --- Instructions and Start Screen for current letter ---
            Screen('TextSize', window, 40);
            DrawFormattedText(window, 'Press SPACE to spell the next letter', 'center', 'center', textColor);
            Screen('Flip', window);
            
            while true % Wait for spacebar
                [keyIsDown, ~, keyCode] = KbCheck;
                if keyIsDown
                    if keyCode(spaceKey), break;
                    elseif keyCode(escapeKey), sca; close(h); ShowCursor; ListenChar(0); return;
                    end
                end
            end

            % --- Generate Flash Sequence ---
            baseSequence = 1:numSequences;
            flashSequence = repmat(baseSequence, 1, ceil(totalFlashes / numSequences));
            shuffledIndices = randperm(length(flashSequence));
            flashSequence = flashSequence(shuffledIndices);
            flashSequence = flashSequence(1:totalFlashes);
            
            % --- Flashing Loop ---
            fprintf('Starting P300 flashing sequence for letter %d of %d...\n', letterIdx, length(currentTargetWord));
            Screen('TextSize', window, 80);
            WaitSecs(preStreamMs / 1000);
            startRowForTrial = lastReadRow;

            for flashNum = 1:totalFlashes
                if KbCheck && keyCode(escapeKey), break; end

                % --- Draw highlighted grid (the "flash") ---
                % First, draw all characters in the base color
                for r = 1:numRows
                    for c = 1:numCols
                        cellCenterX = gridX_start + (c - 0.5) * cellWidth;
                        cellCenterY = gridY_start + (r - 0.5) * cellHeight;
                        DrawFormattedText(window, spellerMatrix{r, c}, 'center', 'center', textColor, [], [], [], [], [], [cellCenterX-cellWidth/2, cellCenterY-cellHeight/2, cellCenterX+cellWidth/2, cellCenterY+cellHeight/2]);
                    end
                end
                % Then, draw the highlighted characters on top in the highlight color
                sequenceIdx = flashSequence(flashNum);
                if sequenceIdx <= numRows
                    rowToHighlight = sequenceIdx;
                    for c = 1:numCols
                        cellCenterX = gridX_start + (c - 0.5) * cellWidth;
                        cellCenterY = gridY_start + (rowToHighlight - 0.5) * cellHeight;
                        DrawFormattedText(window, spellerMatrix{rowToHighlight, c}, 'center', 'center', highlightColor, [], [], [], [], [], [cellCenterX-cellWidth/2, cellCenterY-cellHeight/2, cellCenterX+cellWidth/2, cellCenterY+cellHeight/2]);
                    end
                else
                    colToHighlight = sequenceIdx - numRows;
                    for r = 1:numRows
                        cellCenterX = gridX_start + (colToHighlight - 0.5) * cellWidth;
                        cellCenterY = gridY_start + (r - 0.5) * cellHeight;
                        DrawFormattedText(window, spellerMatrix{r, colToHighlight}, 'center', 'center', highlightColor, [], [], [], [], [], [cellCenterX-cellWidth/2, cellCenterY-cellHeight/2, cellCenterX+cellWidth/2, cellCenterY+cellHeight/2]);
                    end
                end
                
                Screen('Flip', window);
                WaitSecs(flashDuration);

                % --- Draw base grid again for the ISI (the "un-flash") ---
                % This prevents the screen from going blank.
                for r = 1:numRows
                    for c = 1:numCols
                        cellCenterX = gridX_start + (c - 0.5) * cellWidth;
                        cellCenterY = gridY_start + (r - 0.5) * cellHeight;
                        DrawFormattedText(window, spellerMatrix{r, c}, 'center', 'center', textColor, [], [], [], [], [], [cellCenterX-cellWidth/2, cellCenterY-cellHeight/2, cellCenterX+cellWidth/2, cellCenterY+cellHeight/2]);
                    end
                end
                
                Screen('Flip', window);
                WaitSecs(isiDuration);
            end
            disp('Flashing sequence complete.');

            % --- Data Recording and Prediction ---
            WaitSecs(postStreamMs / 1000);
            opts.DataLines = [startRowForTrial + 1, inf];
            trialEEGData = readmatrix(eegCsvPath, opts) .* 18.3 ./ 64;
            
            if size(trialEEGData, 1) > EXPECTED_SAMPLES
                trialEEGData = trialEEGData(1:EXPECTED_SAMPLES, :);
            elseif size(trialEEGData, 1) < EXPECTED_SAMPLES
                padding = zeros(EXPECTED_SAMPLES - size(trialEEGData, 1), 2);
                trialEEGData = [trialEEGData; padding];
            end
            
            trialEEGData = [trialEEGData, trialEEGData(:,2) - trialEEGData(:,1)];
            flatEEGData = trialEEGData';
            flatEEGData = flatEEGData(:)';

            fprintf('Recorded and processed EEG data with size: %d x %d\n', size(trialEEGData));
            
            predictedChar = callP300PythonPredictor(flatEEGData, flashSequence - 1);
            predictedWord = [predictedWord, predictedChar];
            fprintf('Predicted so far: %s\n', predictedWord);
        end % End of letter loop

        % --- Display final result for the word in the console ---
        fprintf('----------------------------------\n');
        fprintf('Final Prediction for "%s": %s\n', currentTargetWord, predictedWord);
        fprintf('----------------------------------\n');
        
        if wordIdx < length(wordsToSpell)
            DrawFormattedText(window, 'Word Complete! Press SPACE for the next word.', 'center', 'center', textColor);
        else
            DrawFormattedText(window, 'All words finished! Press ESC to exit.', 'center', 'center', textColor);
        end
        Screen('Flip', window);

        while true % Wait for space or escape
            [keyIsDown, ~, keyCode] = KbCheck;
            if keyIsDown
                if keyCode(spaceKey) && wordIdx < length(wordsToSpell)
                    break;
                elseif keyCode(escapeKey)
                    sca; close(h); ShowCursor; ListenChar(0); return;
                end
            end
        end

    end % End of word loop

    % --- Cleanup ---
    sca;
    close(h);
    ShowCursor;
    ListenChar(0);
    disp('GUI closed.');

catch
    % If an error occurs, make sure to close the screen and show the cursor.
    sca;
    ShowCursor;
    ListenChar(0);
    psychrethrow(psychlasterror);
end

% --- Helper Function (replicated from data collection script) ---
function [cNoisez, cbaseline, cidx] = updateEEGPlotAndNoise(cplotdata, EEGplot, titletext, FreqWindow, SAMPLE_FREQ, timeWindow, NoiseMU, NoiseSigma, cidx, Zf1, tempdata1, cbaseline)
    cdata = cplotdata(:,[1 2]);
    [pxx, ~] = pwelch(cdata, SAMPLE_FREQ*timeWindow, 0, FreqWindow, SAMPLE_FREQ);
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
