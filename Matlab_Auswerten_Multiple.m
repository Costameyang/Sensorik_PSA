%% 
% ---------------------------------------------------------
% Vergleich von Range- und Doppler-Profilen aus mehreren Dateien
% ---------------------------------------------------------
clear; clc;

% 1. HIER DATEINAMEN EINTRAGEN
% Beispiel: filenames = {'profile_data_1.mat', 'profile_data_2.mat'};
filenames = {'Task3_RangeProfile_Detection0_RadarCube8.npy_fact_300_Rang_hanning_Dopp_hanning.mat', 'Task3_RangeProfile_Detection0_RadarCube8.npy_fact_300_Rang_boxcar_Dopp_hanning.mat'}; 

% Definition der Farben für die Dateien (1=Rot, 2=Blau, 3=Grün)
file_colors = {'r', 'b', 'g', 'k', 'm'}; % Erweitert für Sicherheit

% 2. Legende an- oder ausschalten (true = AN, false = AUS)
show_legend = false;

% Figur erstellen
figure('Name', 'Profile Comparison', 'Color', 'w', 'Position', [100, 100, 1200, 900]);

% Loop über alle Dateien
for i = 1:length(filenames)
    fn = filenames{i};
    col = file_colors{i};
    
    if ~isfile(fn)
        warning('Datei "%s" nicht gefunden. Überspringe...', fn);
        continue;
    end
    
    D = load(fn);
    
    % Name für Legende (Dateiname ohne Endung)
    [~, name_no_ext, ~] = fileparts(fn);
    leg_prefix = strrep(name_no_ext, '_', '\_'); % Unterstriche für Titel escapen
    
    % =========================================================
    % PLOT 1: Oben (Range Profile)
    % =========================================================
    subplot(2, 1, 1);
    hold on; grid on; box on;
    
    % 1. Signal
    if isfield(D, 'range_profile_db')
        plot(D.range_axis, D.range_profile_db, ...
            'Color', col, 'LineStyle', '-', 'LineWidth', 1.5, ...
            'DisplayName', [leg_prefix ' - Signal']);
    end
    
    % 2. Threshold
    if isfield(D, 'range_threshold_db') && ~isempty(D.range_threshold_db)
        plot(D.range_axis, D.range_threshold_db, ...
            'Color', col, 'LineStyle', '--', 'LineWidth', 1.2, ...
            'DisplayName', [leg_prefix ' - Threshold']);
    end
    
    % 3. Selektierte Range (Vertikale Linie)
    if isfield(D, 'selected_range_m')
        xline(D.selected_range_m, ':', 'Color', col, 'LineWidth', 1.5, ...
              'DisplayName', [leg_prefix ' - Selected']);
    end
    
    % 4. Detektionen
    if isfield(D, 'range_detection_marker_indices') && ~isempty(D.range_detection_marker_indices)
        idx = double(D.range_detection_marker_indices) + 1;
        valid_idx = idx(idx <= length(D.range_axis));
        if ~isempty(valid_idx)
            plot(D.range_axis(valid_idx), D.range_profile_db(valid_idx), ...
                'o', 'Color', col, 'MarkerFaceColor', col, ... % Gleiche Farbe gefüllt
                'MarkerSize', 6, 'DisplayName', [leg_prefix ' - Detections']);
        end
    end

    % =========================================================
    % PLOT 2: Unten (Doppler Profile)
    % =========================================================
    subplot(2, 1, 2);
    hold on; grid on; box on;
    
    % 1. Signal
    if isfield(D, 'doppler_profile_db')
        plot(D.velocity_axis, D.doppler_profile_db, ...
            'Color', col, 'LineStyle', '-', 'LineWidth', 1.5, ...
            'DisplayName', [leg_prefix ' - Signal']);
    end
    
    % 2. Threshold
    if isfield(D, 'doppler_threshold_db') && ~isempty(D.doppler_threshold_db)
        plot(D.velocity_axis, D.doppler_threshold_db, ...
            'Color', col, 'LineStyle', '--', 'LineWidth', 1.2, ...
            'DisplayName', [leg_prefix ' - Threshold']);
    end
    
    % 3. Selektierte Velocity (Vertikale Linie)
    if isfield(D, 'selected_velocity_ms')
        xline(D.selected_velocity_ms, ':', 'Color', col, 'LineWidth', 1.5, ...
              'DisplayName', [leg_prefix ' - Selected']);
    end
    
    % 4. Detektionen
    if isfield(D, 'doppler_detection_marker_indices') && ~isempty(D.doppler_detection_marker_indices)
        idx = double(D.doppler_detection_marker_indices) + 1;
        valid_idx = idx(idx <= length(D.velocity_axis));
        if ~isempty(valid_idx)
            plot(D.velocity_axis(valid_idx), D.doppler_profile_db(valid_idx), ...
                'o', 'Color', col, 'MarkerFaceColor', col, ...
                'MarkerSize', 6, 'DisplayName', [leg_prefix ' - Detections']);
        end
    end
end

% Finish Subplot 1 (Range)
subplot(2, 1, 1);
xlabel('Range (m)');
ylabel('Power (dB)');
title('Range Profile Comparison');
axis tight; 
% Legende nur für Signale anzeigen (sonst wird es zu voll), oder 'best' für alles

if show_legend == true
    legend('show', 'Location', 'best');
else
    legend('off');
end

% Finish Subplot 2 (Doppler)
subplot(2, 1, 2);
xlabel('Velocity (m/s)');
ylabel('Power (dB)');
title('Doppler Profile Comparison');
axis tight;
if show_legend == true
    legend('show', 'Location', 'best');
else
    legend('off');
end