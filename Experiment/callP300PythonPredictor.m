% This script runs a P300 speller experiment by flashing rows and columns
% of a 6x6 matrix in a random sequence.

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

    % --- Screen Setup ---
    Screen('Preference', 'SkipSyncTests', 1);
    screens = Screen('Screens');
    screenNumber = max(screens);

    % Define colors
    backgroundColor = [128 128 128]; % Grey
    textColor = [255 255 255];       % White
    highlightColor = [0 255 0];      % Green for highlighted characters

    % Open a window
    [window, ~] = Screen('OpenWindow', screenNumber, backgroundColor);
    [screenXpixels, screenYpixels] = Screen('WindowSize', window);

    % Set text properties
    Screen('TextFont', window, 'Courier New');
    Screen('TextSize', window, 80);
    Screen('TextStyle', window, 1); % Bold text

    % --- Speller Matrix Definition ---
    spellerMatrix = {
        'A', 'B', 'C', 'D', 'E', 'F';
        'G', 'H', 'I', 'J', 'K', 'L';
        'M', 'N', 'O', 'P', 'Q', 'R';
        'S', 'T', 'U', 'V', 'W', 'X';
        'Y', 'Z', '0', '1', '2', '3';
        '4', '5', '6', '7', '8', '9'
    };

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

    % --- Instructions and Start Screen ---
    Screen('TextSize', window, 40);
    DrawFormattedText(window, 'Press SPACE to start the trial', 'center', 'center', textColor);
    Screen('Flip', window);
    
    % Wait for spacebar press to begin
    while true
        [keyIsDown, ~, keyCode] = KbCheck;
        if keyIsDown
            if keyCode(spaceKey)
                break;
            elseif keyCode(escapeKey)
                sca; ShowCursor; ListenChar(0); return;
            end
        end
    end
    
    % --- Flashing Loop ---
    disp('Starting P300 flashing sequence...');
    Screen('TextSize', window, 80); % Reset text size for speller

    for flashNum = 1:totalFlashes
        % Check for ESC key press to abort the trial
        [keyIsDown, ~, keyCode] = KbCheck;
        if keyIsDown && keyCode(escapeKey)
            break;
        end

        % --- Draw the base grid (all white) ---
        for r = 1:numRows
            for c = 1:numCols
                cellCenterX = gridX_start + (c - 0.5) * cellWidth;
                cellCenterY = gridY_start + (r - 0.5) * cellHeight;
                DrawFormattedText(window, spellerMatrix{r, c}, 'center', 'center', textColor, [], [], [], [], [], [cellCenterX-cellWidth/2, cellCenterY-cellHeight/2, cellCenterX+cellWidth/2, cellCenterY+cellHeight/2]);
            end
        end

        % --- Highlight the current row or column ---
        sequenceIdx = flashSequence(flashNum);
        
        if sequenceIdx <= numRows % It's a row
            rowToHighlight = sequenceIdx;
            for c = 1:numCols
                cellCenterX = gridX_start + (c - 0.5) * cellWidth;
                cellCenterY = gridY_start + (rowToHighlight - 0.5) * cellHeight;
                DrawFormattedText(window, spellerMatrix{rowToHighlight, c}, 'center', 'center', highlightColor, [], [], [], [], [], [cellCenterX-cellWidth/2, cellCenterY-cellHeight/2, cellCenterX+cellWidth/2, cellCenterY+cellHeight/2]);
            end
        else % It's a column
            colToHighlight = sequenceIdx - numRows;
            for r = 1:numRows
                cellCenterX = gridX_start + (colToHighlight - 0.5) * cellWidth;
                cellCenterY = gridY_start + (r - 0.5) * cellHeight;
                DrawFormattedText(window, spellerMatrix{r, colToHighlight}, 'center', 'center', highlightColor, [], [], [], [], [], [cellCenterX-cellWidth/2, cellCenterY-cellHeight/2, cellCenterX+cellWidth/2, cellCenterY+cellHeight/2]);
            end
        end
        
        % Show the highlighted grid (the flash)
        Screen('Flip', window);
        WaitSecs(flashDuration);

        % --- Draw the base grid again to "un-highlight" ---
        for r = 1:numRows
            for c = 1:numCols
                cellCenterX = gridX_start + (c - 0.5) * cellWidth;
                cellCenterY = gridY_start + (r - 0.5) * cellHeight;
                DrawFormattedText(window, spellerMatrix{r, c}, 'center', 'center', textColor, [], [], [], [], [], [cellCenterX-cellWidth/2, cellCenterY-cellHeight/2, cellCenterX+cellWidth/2, cellCenterY+cellHeight/2]);
            end
        end
        
        % Show the un-highlighted grid
        Screen('Flip', window);
        WaitSecs(isiDuration);
    end

    % --- End of Trial ---
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
