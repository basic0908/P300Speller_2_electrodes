% This script initializes a Psychtoolbox window and displays a static
% 6x6 P300 speller matrix with characters A-Z and 0-9.
% The window can be closed by pressing the 'ESCAPE' key.

% --- Cleanup and Assertions ---
% Close any open screens and clear variables.
sca;
close all;
clearvars;

% Unify key names across different operating systems
KbName('UnifyKeyNames');
escapeKey = KbName('ESCAPE');

try
    % --- Screen Setup ---
    % Basic setup for Psychtoolbox screen
    Screen('Preference', 'SkipSyncTests', 1);
    screens = Screen('Screens');
    screenNumber = max(screens); % Use the main screen

    % Define colors
    backgroundColor = [128 128 128]; % Grey
    textColor = [255 255 255];       % White

    % Open a window
    [window, ~] = Screen('OpenWindow', screenNumber, backgroundColor);
    [screenXpixels, screenYpixels] = Screen('WindowSize', window);

    % Set text properties
    Screen('TextFont', window, 'Courier New');
    Screen('TextSize', window, 80); % Increased font size for characters
    Screen('TextStyle', window, 1);   % Bold text

    % --- Speller Matrix Definition ---
    spellerMatrix = {
        'A', 'B', 'C', 'D', 'E', 'F';
        'G', 'H', 'I', 'J', 'K', 'L';
        'M', 'N', 'O', 'P', 'Q', 'R';
        'S', 'T', 'U', 'V', 'W', 'X';
        'Y', 'Z', '0', '1', '2', '3';
        '4', '5', '6', '7', '8', '9'
    };

    [rows, cols] = size(spellerMatrix);

    % --- Grid Layout Calculation ---
    % Define the overall size and position of the grid
    gridWidth = screenXpixels * 0.8;
    gridHeight = screenYpixels * 0.8;
    
    % Center the grid on the screen
    gridX_start = (screenXpixels - gridWidth) / 2;
    gridY_start = (screenYpixels - gridHeight) / 2;

    % Calculate the size of each cell
    cellWidth = gridWidth / cols;
    cellHeight = gridHeight / rows;

    % --- Initial Drawing ---
    % Draw all characters in their positions
    for r = 1:rows
        for c = 1:cols
            % Calculate the center of the current cell
            cellCenterX = gridX_start + (c - 0.5) * cellWidth;
            cellCenterY = gridY_start + (r - 0.5) * cellHeight;
            
            % Draw the character centered in the cell
            DrawFormattedText(window, spellerMatrix{r, c}, 'center', 'center', textColor, [], [], [], [], [], [cellCenterX-cellWidth/2, cellCenterY-cellHeight/2, cellCenterX+cellWidth/2, cellCenterY+cellHeight/2]);
        end
    end
    
    % Show the initial grid
    Screen('Flip', window);
    
    % --- Main Loop ---
    % Wait for the user to press the escape key.
    disp('P300 Speller GUI is running. Press ESC to exit.');
    while true
        [keyIsDown, ~, keyCode] = KbCheck;
        if keyIsDown
            if keyCode(escapeKey)
                break; % Exit the loop if ESC is pressed
            end
        end
        % We don't need to redraw in this loop since the grid is static.
        % A small pause prevents the loop from consuming 100% CPU.
        WaitSecs(0.01); 
    end

    % --- Cleanup ---
    sca; % Closes all screens
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
