function predictedChar = callP300PythonPredictor(eegData, flashSequence)
    % This function calls the Python P300 prediction script.
    % It MUST be saved as a function file, with this 'function' line at the top.
    %
    % Inputs:
    %   eegData (double array): A 1D array of flattened raw EEG data.
    %   flashSequence (double array): A 1D array of flash indices (0-11).
    %
    % Output:
    %   predictedChar (char): The single character predicted by the model.
    %
    % IMPORTANT: Before calling this, ensure the Python environment is set
    % in the MATLAB Command Window, e.g.:
    % >> pyenv('Version', 'path/to/your/p300env_py39/Scripts/python.exe');

    % --- 1. Define Paths ---
    pythonScriptPath = 'C:\Users\ryoii\OneDrive\Documents\GitHub\P300Speller_2_electrodes\Python';
    pythonModuleName = 'predict_p300';
    modelPath = 'C:\Users\ryoii\OneDrive\Documents\GitHub\P300Speller_2_electrodes\models\p300_classifier_model.pkl';
    featuresPath = 'C:\Users\ryoii\OneDrive\Documents\GitHub\P300Speller_2_electrodes\models\p300_feature_selection_indices.pkl';

    % --- 2. Call the Python Function ---
    try
        % Add the Python script's directory to the Python path
        if count(py.sys.path, pythonScriptPath) == 0
            insert(py.sys.path, int32(0), pythonScriptPath);
        end

        % Import the Python module
        pyModule = py.importlib.import_module(pythonModuleName);
        py.importlib.reload(pyModule); % Reload to get the latest changes

        % Convert MATLAB arrays to Python-compatible types
        eegDataPy = py.list(eegData);
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
        predictedChar = 'Error';
    end
end
