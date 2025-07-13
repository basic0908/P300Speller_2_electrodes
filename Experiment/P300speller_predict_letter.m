function predictedChar = callP300PythonPredictor()
    % This function simulates EEG data and a flash sequence for a P300 speller,
    % then calls a Python script to predict the target character.
    %
    % IMPORTANT: Before running this function, you MUST set the Python environment
    % in the MATLAB Command Window. Restart MATLAB and run:
    % >> pyenv('Version', 'C:\Users\ryoii\OneDrive\Documents\GitHub\P300Speller_2_electrodes\p300env_py39\Scripts\python.exe');
    %
    % You can confirm the loaded environment by typing 'pyenv' in the command window.

    % --- 1. Define Paths and Parameters ---
    % IMPORTANT: Update this path to the actual location of your Python script.
    pythonScriptPath = 'C:\Users\ryoii\OneDrive\Documents\GitHub\P300Speller_2_electrodes\Python';
    pythonModuleName = 'predict_p300'; % The name of your .py file without the extension
    
    % CORRECTED: Paths to the model files now point to the correct 'models' directory.
    modelPath = 'C:\Users\ryoii\OneDrive\Documents\GitHub\P300Speller_2_electrodes\models\p300_classifier_model.pkl';
    featuresPath = 'C:\Users\ryoii\OneDrive\Documents\GitHub\P300Speller_2_electrodes\models\p300_feature_selection_indices.pkl';

    % Experiment parameters
    numChannels = 3;
    numSamples = 9720;
    numSequences = 12; % 6 rows + 6 columns
    totalFlashes = 150;

    % --- 2. Generate Flash Sequence ---
    % This sequence determines which row/column flashes at each step.
    % In a real experiment, this would be the sequence you present to the user.
    % The indices are 0-5 for rows and 6-11 for columns.
    repsPerSequence = totalFlashes / numSequences; % Should be 12.5 for 150 flashes
    baseSequence = 0:(numSequences-1);
    
    % Create a sequence with approximately equal repetitions
    flashSequence = repmat(baseSequence, 1, ceil(repsPerSequence));
    
    % Randomize the order and truncate to exactly 150 flashes
    shuffledIndices = randperm(length(flashSequence));
    flashSequence = flashSequence(shuffledIndices);
    flashSequence = flashSequence(1:totalFlashes);
    
    fprintf('Generated a flash sequence of %d flashes.\n', length(flashSequence));

    % --- 3. Simulate EEG Data ---
    % This creates random data as a placeholder for your actual EEG recordings.
    % The data is flattened into a 1D array, as expected by the Python function.
    simulatedEEGData = randn(1, numChannels * numSamples);
    fprintf('Generated simulated EEG data with %d total points.\n', length(simulatedEEGData));

    % --- 4. Call the Python Function ---
    try
        % Add the Python script's directory to the Python path
        if count(py.sys.path, pythonScriptPath) == 0
            insert(py.sys.path, int32(0), pythonScriptPath);
        end

        % Import the Python module
        pyModule = py.importlib.import_module(pythonModuleName);
        py.importlib.reload(pyModule); % Reload to get the latest changes

        % Convert MATLAB arrays to Python-compatible types (lists)
        eegDataPy = py.list(simulatedEEGData);
        flashSequencePy = py.list(int64(flashSequence)); % Ensure integers

        % Call the function
        fprintf('Calling Python function predict_speller_char...\n');
        pyResult = pyModule.predict_speller_char(eegDataPy, flashSequencePy, modelPath, featuresPath);
        
        % Convert the Python string result back to a MATLAB char array
        predictedChar = char(pyResult);

    catch ME
        disp('An error occurred while calling the Python function:');
        disp(ME.message);
        % Display detailed Python error if available
        if ~isempty(ME.cause) && isa(ME.cause{1}, 'py.Exception.BaseException')
            disp('--- Python Traceback ---');
            disp(char(py.traceback.format_exc()));
        end
        predictedChar = 'Error: Python Call';
    end

    % --- 5. Display the Result ---
    fprintf('\n==================================\n');
    fprintf('Predicted Character: %s\n', predictedChar);
    fprintf('==================================\n');

end
