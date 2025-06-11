clc;clear all;
try
    % Setup Psychtoolbox
    Screen('Preference', 'SkipSyncTests', 1); % Disable sync test for testing
    [win, winRect] = Screen('OpenWindow', max(Screen('Screens')), 0); % black background
    Screen('TextSize', win, 64);  % Set font size

    % Define characters
    chars = ['A', 'B', 'C', 'D', 'E', 'F', 'G'];
    chars = chars(randperm(length(chars)));  % Randomize order

    % Display each character for 2 seconds
    for i = 1:length(chars)
        DrawFormattedText(win, chars(i), 'center', 'center', [255 255 255]);  % White text
        Screen('Flip', win);  % Show character
        WaitSecs(2);  % Wait 2 seconds
        Screen('Flip', win);  % Clear screen
        WaitSecs(0.2);  % Optional short blank interval
    end

    % End screen
    DrawFormattedText(win, 'Done.', 'center', 'center', [255 255 255]);
    Screen('Flip', win);
    WaitSecs(2);

    % Close screen
    Screen('CloseAll');
catch ME
    Screen('CloseAll');
    rethrow(ME);
end
