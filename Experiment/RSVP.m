%{
TODO
- 日本語コメント修正
- 日本語の指示コマンド入れ替え
- プロトコルチェック
%}
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
RSVP実験フロー
=========================
1. 被験者ごとに合計40試行のRSVPタスクを実施します。
2. 各試行では以下を行います：
   a. 同じ位置に21文字の文字列を高速で提示します。
      - ターゲット文字が1つ（緑色）
      - 非ターゲット文字が20個（白色）
      - 文字の順序はランダムに並べ替えられます
   b. 提示レートは10Hz（各文字の提示時間は100ミリ秒）
3. 文字列の提示終了後：
   a. 5秒間の応答ウィンドウで、参加者がターゲット文字をキーボードで入力します。
4. 正誤のフィードバックは行いません。
5. 正答率（T1%）で成績を評価します：
   - 40試行中、ターゲットを正しく識別できた割合（％）
%}

clc;
clear;

try
    % === Psychtoolbox Initialization ===
    Screen('Preference', 'SkipSyncTests', 1);
    screens = Screen('Screens');
    screenNumber = max(screens);
    [win, winRect] = Screen('OpenWindow', screenNumber, 0);  % black background
    Screen('TextSize', win, 64);
    HideCursor;

    % === Parameters ===
    all_chars = ['A':'Z'];
    n_trials = 40;
    stream_len = 21;
    stim_duration = 0.1;  % 100 ms per character
    response_window = 5;  % 5 seconds

    % === Result Storage ===
    target_chars = cell(1, n_trials);
    responses = cell(1, n_trials);

    % === Instruction Screen ===
    DrawFormattedText(win, 'RSVP Task\n\nCharacters will be flashed.\nRemember the green letter.\n\nPress Enter to begin.\nPress Escape anytime to quit.', 'center', 'center', [255 255 255]);
    Screen('Flip', win);

    % === Wait for Enter or Escape ===
    while true
        [keyIsDown, ~, keyCode] = KbCheck;
        if keyIsDown
            keyName = KbName(find(keyCode, 1));
            if ischar(keyName)
                if any(strcmpi(keyName, {'return', 'kp_enter'}))
                    break;
                elseif any(strcmpi(keyName, {'escape', 'esc'}))
                    error('日本語行ける？');
                end
            end
        end
    end

    % === Trial Loop ===
    for trial = 1:n_trials
        % Select random target
        target = all_chars(randi(length(all_chars)));
        distractors = all_chars(all_chars ~= target);
        distractors = distractors(randperm(length(distractors), stream_len - 1));
        stream = [target, distractors];
        stream = stream(randperm(length(stream)));  % Shuffle stream

        target_chars{trial} = target;

        % Present each character at 10 Hz
        for i = 1:stream_len
            if stream(i) == target
                color = [0 255 0];  % Green target
            else
                color = [255 255 255];  % White non-target
            end
            DrawFormattedText(win, stream(i), 'center', 'center', color);
            Screen('Flip', win);
            WaitSecs(stim_duration);
        end

        % Clear screen
        Screen('Flip', win);

        % Prompt user for keyboard response
        DrawFormattedText(win, 'Type the target character on the keyboard.', 'center', 'center', [255 255 255]);
        Screen('Flip', win);

        resp = '';
        start_time = GetSecs;
        while GetSecs - start_time < response_window
            [keyIsDown, ~, keyCode] = KbCheck;
            if keyIsDown
                keyPressed = KbName(find(keyCode, 1));
                if ischar(keyPressed)
                    if any(strcmpi(keyPressed, {'escape', 'esc'}))
                        error('日本語行ける？');
                    elseif length(keyPressed) == 1 && any(upper(keyPressed) == all_chars)
                        resp = upper(keyPressed);
                        break;
                    end
                end
            end
        end
        responses{trial} = resp;

        % Inter-trial blank screen
        Screen('Flip', win);
        WaitSecs(1);
    end

    % === Accuracy Calculation ===
    correct = 0;
    for i = 1:n_trials
        if strcmpi(target_chars{i}, responses{i})
            correct = correct + 1;
        end
    end
    t1_accuracy = (correct / n_trials) * 100;

    % === Show Results ===
    msg = sprintf('Experiment Done.\n\nCorrect: %d / %d\nT1 Accuracy: %.2f%%', correct, n_trials, t1_accuracy);
    DrawFormattedText(win, msg, 'center', 'center', [255 255 255]);
    Screen('Flip', win);
    WaitSecs(5);

    % === Cleanup ===
    Screen('CloseAll');
    ShowCursor;

    % === Print to Console ===
    fprintf('Target characters:\n%s\n', strjoin(target_chars, ', '));
    fprintf('Responses:\n%s\n', strjoin(responses, ', '));
    fprintf('T1 Accuracy: %.2f%%\n', t1_accuracy);

catch ME
    Screen('CloseAll');
    ShowCursor;
    fprintf('Error: %s\n', ME.message);
end
