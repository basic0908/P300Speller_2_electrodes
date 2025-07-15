% This script runs a P300 speller experiment by flashing rows and columns
% of a 6x6 matrix, while displaying a live EEG plot.

% --- Cleanup and Assertions ---
sca;
close all;
clearvars;

% Unify key names across different operating systems
KbName('UnifyKeyNames');
escapeKey = KbName('ESCAPE');
spaceKey = KbName('space');

try
    % --- Experiment Parameters ---
    flashDuration = 0.1; % 100 ms
    isiDuration = 0.05;  % 50 ms Inter-Stimulus Interval
    totalFlashes = 150;
    numRows = 6;
    numCols = 6;
    numSequences = numRows + numCols; % 12 total sequences

    % --- EEG Parameters (replicated from data collection script) ---
    eegCsvPath = "C:\Users\ryoii\OneDrive\Documents\GitHub\P300Speller_2_electrodes\VIERecorder\VieOutput\VieRawData.csv";
    opts = detectImportOptions(eegCsvPath);
    opts.SelectedVariableNames = [2 3]; % Select the two EEG channels
    SAMPLE_FREQ = 600;
    
    % Plotting window parameters
    plotTimeWindow = 4; % seconds
    plotDataBuffer = repmat(0, SAMPLE_FREQ * plotTimeWindow, 3);
    plotIndex = 1;
    plotBaseline = [0 0 0];
    
    % Filter setup
    [Bf, Af] = butter(4, [3/(SAMPLE_FREQ/2), 40/(SAMPLE_FREQ/2)]);
    Zf = [];
    
    % Initialize the row counter for the CSV file
    % This reads the initial size of the file to avoid re-reading old data
    try
        initialData = readmatrix(eegCsvPath, opts);
        lastReadRow = size(initialData, 1);
    catch
        lastReadRow = 0; % If file doesn't exist or is empty
    end
    
    % Noise parameters (as in the reference script)
    NoiseMU = [138.5023,138.5023,12.1827,12.1827,103.387,103.387,0.0353,0.0353,10.124,10.124];
    NoiseSigma = [123.2406,123.2406,6.0254,6.0254,69.4416,69.4416,0.0039,0.0039,2.899,2.899];

    % --- Screen Setup ---
    Screen('Preference', 'SkipSyncTests', 1);
    screens = Screen('Screens');
    screenNumber = max(screens);

    % Define colors
    backgroundColor = [0 0 0];         % Black
    textColor = [128 128 128];         % Grey
    highlightColor = [255 255 255];    % White for highlighted characters

    % Open a window
    [window, ~] = Screen('OpenWindow', screenNumber, backgroundColor);
    [screenXpixels, screenYpixels] = Screen('WindowSize', window);
    ListenChar(2); % Suppress keyboard input to MATLAB command window
    HideCursor;

    % Set text properties
    Screen('TextFont', window, 'Courier New');
    Screen('TextSize', window, 80);
    Screen('TextStyle', window, 1); % Bold text

    % --- EEG Plot Window Setup (replicated exactly) ---
    windowsize = get(0,'MonitorPositions');
    h = figure('Position', [windowsize(1,1), windowsize(1,2), windowsize(1,3), 200], 'Color', 'k');
    h.MenuBar = 'none';
    h.ToolBar = 'none';
    EEGplot = plot( (1:SAMPLE_FREQ*plotTimeWindow)/SAMPLE_FREQ, plotDataBuffer - plotBaseline, 'LineWidth', 1);
    ha1 = gca;
    ha1.GridColor = [1 1 1];
    set(gca, 'Color', 'k');
    h_yaxis = ha1.YAxis; h_yaxis.Color = 'w'; h_yaxis.Label.Color = [1 1 1];
    h_xaxis = ha1.XAxis; h_xaxis.Color = 'w'; h_xaxis.Label.Color = [1 1 1];
    xlim([0 plotTimeWindow]); ylim([-75 75]);
    titletext = title('EEG (0s)', 'Color', 'w', 'FontSize', 22);
    xlabel('time (s)'); ylabel('\muV'); yline(0, 'w--');
    lgd = legend(EEGplot, {'L', 'R', 'diff'}); lgd.TextColor = [1 1 1];

    % --- Speller Matrix Definition ---
    spellerMatrix = {
        'A', 'B', 'C', 'D', 'E', 'F';
        'G', 'H', 'I', 'J', 'K', 'L';
        'M', 'N', 'O', 'P', 'Q', 'R';
        'S', 'T', 'U', 'V', 'W', 'X';
        'Y', 'Z', '0', '1', '2', '3';
        '4', '5', '6', '7', '8', '9'
    };
    [numRows, numCols] = size(spellerMatrix);

    % --- Grid Layout Calculation ---
    gridWidth = screenXpixels * 0.8;
    gridHeight = screenYpixels * 0.8;
    gridX_start = (screenXpixels - gridWidth) / 2;
    gridY_start = (screenYpixels - gridHeight) / 2;
    cellWidth = gridWidth / numCols;
    cellHeight = gridHeight / numRows;

    % --- Generate Flash Sequence ---
    repsPerSequence = totalFlashes / numSequences;
    baseSequence = 1:numSequences;
    flashSequence = repmat(baseSequence, 1, ceil(repsPerSequence));
    shuffledIndices = randperm(length(flashSequence));
    flashSequence = flashSequence(shuffledIndices);
    flashSequence = flashSequence(1:totalFlashes);

    % --- Instructions and Start Screen (with live EEG plot) ---
    Screen('TextSize', window, 40);
    DrawFormattedText(window, 'Press SPACE to start the trial', 'center', 'center', textColor);
    Screen('Flip', window);
    
    while true
        % --- Update EEG Plot ---
        opts.DataLines = [lastReadRow + 1, inf];
        try
            newData = readmatrix(eegCsvPath, opts) .* 18.3 ./ 64;
            if ~isempty(newData)
                newData = [newData, newData(:,2) - newData(:,1)]; % Add diff channel
                lastReadRow = lastReadRow + size(newData,1);
                [filteredData, Zf] = filter(Bf, Af, newData, Zf);
                
                % Update circular plot buffer
                if plotIndex + size(newData,1) - 1 < SAMPLE_FREQ * plotTimeWindow
                    plotDataBuffer(plotIndex : plotIndex + size(newData,1) - 1, :) = filteredData;
                    plotIndex = plotIndex + size(newData,1);
                else
                    plotDataBuffer(plotIndex:end, :) = filteredData(1:size(plotDataBuffer,1) - plotIndex + 1, :);
                    plotIndex = 1;
                    plotBaseline = nanmean(plotDataBuffer);
                end
                
                % Update the plot using the helper function
                updateEEGPlotAndNoise(plotDataBuffer, EEGplot, titletext, 4:0.5:40, SAMPLE_FREQ, plotTimeWindow, NoiseMU, NoiseSigma, plotIndex, Zf, filteredData, plotBaseline);
            end
        catch
            % Ignore if file is temporarily locked or empty
        end
        
        % --- Check for Key Press ---
        [keyIsDown, ~, keyCode] = KbCheck;
        if keyIsDown
            if keyCode(spaceKey)
                break;
            elseif keyCode(escapeKey)
                sca; close(h); ShowCursor; ListenChar(0); return;
            end
        end
    end
    
    % --- Flashing Loop ---
    disp('Starting P300 flashing sequence...');
    Screen('TextSize', window, 80); % Reset text size for speller
    
    startRowForTrial = lastReadRow; % Record the start row for this trial's data

    for flashNum = 1:totalFlashes
        [keyIsDown, ~, keyCode] = KbCheck;
        if keyIsDown && keyCode(escapeKey), break; end

        % Draw base grid
        for r = 1:numRows
            for c = 1:numCols
                cellCenterX = gridX_start + (c - 0.5) * cellWidth;
                cellCenterY = gridY_start + (r - 0.5) * cellHeight;
                DrawFormattedText(window, spellerMatrix{r, c}, 'center', 'center', textColor, [], [], [], [], [], [cellCenterX-cellWidth/2, cellCenterY-cellHeight/2, cellCenterX+cellWidth/2, cellCenterY+cellHeight/2]);
            end
        end

        % Highlight current row/column
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

        % Draw base grid again (un-highlight)
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

    % --- End of Trial ---
    % Read the EEG data for the entire trial epoch
    WaitSecs(1.0); % Post-stream wait
    opts.DataLines = [startRowForTrial + 1, inf];
    trialEEGData = readmatrix(eegCsvPath, opts) .* 18.3 ./ 64;
    trialEEGData = [trialEEGData, trialEEGData(:,2) - trialEEGData(:,1)];
    
    % For now, just display the size to confirm it was read
    fprintf('Recorded EEG data for trial with size: %d x %d\n', size(trialEEGData, 1), size(trialEEGData, 2));

    Screen('TextSize', window, 40);
    DrawFormattedText(window, 'Trial Finished. Press ESC to exit.', 'center', 'center', textColor);
    Screen('Flip', window);
    disp('Flashing sequence complete.');

    while true
        [keyIsDown, ~, keyCode] = KbCheck;
        if keyIsDown && keyCode(escapeKey)
            break;
        end
        WaitSecs(0.01);
    end

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
