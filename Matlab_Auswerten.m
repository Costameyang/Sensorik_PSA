% Datei laden (Dateinamen ggf. anpassen)
load('Task3_RangeProfile_Detection0_RadarCube8.npy.mat'); 

figure('Color', 'w', 'Position', [100, 100, 1000, 800]);

% --- 1. Oberer Plot: Range Profile ---
subplot(2, 1, 1);
hold on; grid on; box on;

% Signal plotten
plot(range_axis, range_profile_db, 'b-', 'LineWidth', 1.5, 'DisplayName', 'Signal');

% Threshold plotten (falls vorhanden)
if exist('range_threshold_db', 'var') && ~isempty(range_threshold_db)
    plot(range_axis, range_threshold_db, 'r-', 'LineWidth', 2, 'DisplayName', 'Threshold');
end

% Rote vertikale Linie (Selected Range)
xline(selected_range_m, '--r', 'LineWidth', 1.5, 'DisplayName', 'Selected Range');

% Gelbe Marker für Detektionen
if exist('range_detection_marker_indices', 'var') && ~isempty(range_detection_marker_indices)
    % MATLAB Indizes sind 1-basiert, Python 0-basiert. Daher +1 rechnen!
    idx = double(range_detection_marker_indices) + 1;
    
    x_vals = range_axis(idx);
    y_vals = range_profile_db(idx);
    
    plot(x_vals, y_vals, 'o', 'MarkerFaceColor', 'yellow', ...
         'MarkerEdgeColor', 'red', 'LineWidth', 1.5, 'MarkerSize', 8, ...
         'DisplayName', 'Detections');
end

xlabel('Range (m)');
ylabel('Power (dB)');
title(sprintf('Range Profile at Velocity = %.3f m/s', selected_velocity_ms));
ylim([50 190]); % Skalierung wie in Python fixiert
legend('show', 'Location', 'northeast');


% --- 2. Unterer Plot: Doppler Profile ---
subplot(2, 1, 2);
hold on; grid on; box on;

% Signal
plot(velocity_axis, doppler_profile_db, 'b-', 'LineWidth', 1.5, 'DisplayName', 'Signal');

% Threshold
if exist('doppler_threshold_db', 'var') && ~isempty(doppler_threshold_db)
    plot(velocity_axis, doppler_threshold_db, 'r-', 'LineWidth', 2, 'DisplayName', 'Threshold');
end

% Rote vertikale Linie (Selected Velocity)
xline(selected_velocity_ms, '--r', 'LineWidth', 1.5, 'DisplayName', 'Selected Vel');

% Gelbe Marker
if exist('doppler_detection_marker_indices', 'var') && ~isempty(doppler_detection_marker_indices)
    idx = double(doppler_detection_marker_indices) + 1; % Python zu MATLAB Index
    
    x_vals = velocity_axis(idx);
    y_vals = doppler_profile_db(idx);
    
    plot(x_vals, y_vals, 'o', 'MarkerFaceColor', 'yellow', ...
         'MarkerEdgeColor', 'red', 'LineWidth', 1.5, 'MarkerSize', 8, ...
         'DisplayName', 'Detections');
end

xlabel('Velocity (m/s)');
ylabel('Power (dB)');
title(sprintf('Doppler Profile at Range = %.2f m', selected_range_m));
ylim([50 190]); 
legend('show', 'Location', 'northeast');