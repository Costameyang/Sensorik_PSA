% ---------------------------------------------------------
% MATLAB Skript für 2 Subplots (Range & Doppler)
% ---------------------------------------------------------
clear; clc;

% 1. WICHTIG: Laden Sie die NEUE Datei aus Schritt 1
filename = 'Task3_RangeProfile_Detection0_RadarCube8.npy.mat'; 

if ~isfile(filename)
    error(['Datei "' filename '" nicht gefunden. ' ...
           'Bitte führen Sie erst "plot_range_profile_at_detection" in Python aus!']);
end

D = load(filename);

% Figur erstellen
figure('Name', 'Detection Profile Analysis', 'Color', 'w', 'Position', [100, 100, 1000, 800]);

% =========================================================
% PLOT 1: Oben (Range Profile)
% =========================================================
subplot(2, 1, 1);
hold on; grid on; box on;

% Signal (Range)
if isfield(D, 'range_profile_db')
    plot(D.range_axis, D.range_profile_db, 'b-', 'LineWidth', 1.5, 'DisplayName', 'Signal');
end

% Threshold (Range)
if isfield(D, 'range_threshold_db') && ~isempty(D.range_threshold_db)
    plot(D.range_axis, D.range_threshold_db, 'r-', 'LineWidth', 2, 'DisplayName', 'Threshold');
end

% Markierung der ausgewählten Position
if isfield(D, 'selected_range_m')
    xline(D.selected_range_m, '--r', 'Selected Range', 'LineWidth', 1.5, 'LabelVerticalAlignment', 'bottom');
end

% Detektionen (Gelbe Punkte)
if isfield(D, 'range_detection_marker_indices') && ~isempty(D.range_detection_marker_indices)
    % Index +1 für Matlab
    idx = double(D.range_detection_marker_indices) + 1;
    valid_idx = idx(idx <= length(D.range_axis));
    
    if ~isempty(valid_idx)
        plot(D.range_axis(valid_idx), D.range_profile_db(valid_idx), 'o', ...
             'MarkerFaceColor', 'yellow', 'MarkerEdgeColor', 'red', ...
             'MarkerSize', 8, 'LineWidth', 1.5, 'DisplayName', 'Detections');
    end
end

title(['Range Profile at Velocity = ' num2str(D.selected_velocity_ms, '%.3f') ' m/s']);
xlabel('Range (m)');
ylabel('Power (dB)');
legend('show', 'Location', 'best');
axis tight; 


% =========================================================
% PLOT 2: Unten (Doppler Profile)
% =========================================================
subplot(2, 1, 2);
hold on; grid on; box on;

% Signal (Doppler)
if isfield(D, 'doppler_profile_db')
    plot(D.velocity_axis, D.doppler_profile_db, 'b-', 'LineWidth', 1.5, 'DisplayName', 'Signal');
end

% Threshold (Doppler)
if isfield(D, 'doppler_threshold_db') && ~isempty(D.doppler_threshold_db)
    plot(D.velocity_axis, D.doppler_threshold_db, 'r-', 'LineWidth', 2, 'DisplayName', 'Threshold');
end

% Markierung der ausgewählten Position
if isfield(D, 'selected_velocity_ms')
    xline(D.selected_velocity_ms, '--r', 'Selected Vel', 'LineWidth', 1.5, 'LabelVerticalAlignment', 'bottom');
end

% Detektionen (Gelbe Punkte)
if isfield(D, 'doppler_detection_marker_indices') && ~isempty(D.doppler_detection_marker_indices)
    % Index +1 für Matlab
    idx = double(D.doppler_detection_marker_indices) + 1;
    valid_idx = idx(idx <= length(D.velocity_axis));
    
    if ~isempty(valid_idx)
        plot(D.velocity_axis(valid_idx), D.doppler_profile_db(valid_idx), 'o', ...
             'MarkerFaceColor', 'yellow', 'MarkerEdgeColor', 'red', ...
             'MarkerSize', 8, 'LineWidth', 1.5, 'DisplayName', 'Detections');
    end
end

title(['Doppler Profile at Range = ' num2str(D.selected_range_m, '%.2f') ' m']);
xlabel('Velocity (m/s)');
ylabel('Power (dB)');
legend('show', 'Location', 'best');
axis tight;