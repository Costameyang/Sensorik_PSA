import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# --- KONFIGURATION ---

# 1. Zuordnung der Spaltennamen zu Kategorien
# Tipp: Du kannst hier mehrere Namen pro Liste eintragen, um verschiedene Länder abzudecken.
CATEGORY_MAPPING = {
    'ICE': ['Benzin', 'Diesel'],
    'EV':  ['Elektro (BEV)'],
    'HV':  ['Full-Hybride (HV)', 'Hybrid (ohne Plug-in)'],
    'PHV': ['Plug-in-Hybrid (PHV)', 'Plug-in-Hybrid'],
    'MHV': ['Mild-Hybrid (MHV)']
    #'FCV': ['Wasserstoff (FCV)'],
    #'N/A': ['N/A']
}

# CATEGORY_MAPPING = {
#     'ICE': ['Benzin', 'Diesel'],
#     'EV':  ['Elektro (BEV)'],
#     'HV':  ['Full-Hybride (HV)', 'Hybrid (ohne Plug-in)', 'Mild-Hybrid (MHV)','Plug-in-Hybrid (PHV)', 'Plug-in-Hybrid'],
#     'NEV': [],
#     'PHV': [],
#     'MHV': [],
#     'FCV': ['Wasserstoff (FCV)'],
#     'N/A': ['N/A']
# }

CATEGORY_MAPPING = {
    'ICE': ['Benzin', 'Diesel'],
    'EV':  [],
    'HV':  [],
    'NEV': ['Elektro (BEV)', 'Full-Hybride (HV)', 'Hybrid (ohne Plug-in)', 'Mild-Hybrid (MHV)','Plug-in-Hybrid (PHV)', 'Plug-in-Hybrid', 'Wasserstoff (FCV)'],
    'PHV': [],
    'MHV': [],
    'FCV': [],
    'N/A': []
}

# 2. Feste Farbzuordnung für die Kategorien
COLOR_MAPPING = {
    'ICE': '#1f77b4',  # Blau
    'EV':  '#2ca02c',  # Grün
    'HV':  '#ff7f0e',  # Orange
    'NEV': '#d62728',  # Rot
    'PHV': '#8c564b',  # Braun
    'MHV': '#9467bd',  # Lila
    'FCV': '#17becf',  # Türkis
    'N/A': '#7f7f7f'   # Grau
}

# --- KONFIGURATION ---

# NEU: Modus umstellen -> 'overlapping' (überlappend) oder 'stacked' (addiert)
PLOT_MODE = 'stacked' 

def process_and_plot(file_path, country_name, mode='overlapping'):
    """
    Erstellt ein Diagramm. Unterstützt 'overlapping' und 'stacked'.
    Prozente beziehen sich immer auf den Anteil am Monats-Gesamtwert.
    """
    if not os.path.exists(file_path):
        print(f"Datei nicht gefunden: {file_path}")
        return

    # 1. Daten laden
    def _load_csv(path):
        for enc in ['utf-8', 'cp1252', 'latin-1']:
            try:
                return pd.read_csv(path, sep=';', encoding=enc)
            except:
                continue
        return pd.read_csv(path, sep=';')

    df = _load_csv(file_path)
    if df.shape[1] > 1:
        df = df.dropna(subset=[df.columns[1]])
    
    # 2. Daten aggregieren
    plot_df = pd.DataFrame()
    time_col = df.columns[0]
    plot_df['Zeit'] = df[time_col].values
    
    active_categories = []
    for category, columns in CATEGORY_MAPPING.items():
        existing_cols = [c for c in columns if c in df.columns]
        if existing_cols:
            series = df[existing_cols].sum(axis=1)
            if series.sum() > 0:
                plot_df[category] = series.values
                active_categories.append(category)

    # Monats-Gesamtsumme für Prozentrechnung
    plot_df['Total_Month'] = plot_df[active_categories].sum(axis=1)

    # 3. Plot erstellen
    plt.figure(figsize=(15, 8))
    
    # Reihenfolge festlegen
    if mode == 'stacked':
        # Von unten nach oben gemäß Mapping-Reihenfolge
        draw_order = [c for c in CATEGORY_MAPPING.keys() if c in active_categories]
        current_stack = np.zeros(len(plot_df))
    else:
        # Overlapping: Größte Fläche nach hinten (Z-Order)
        draw_order = sorted(active_categories, key=lambda c: plot_df[c].mean(), reverse=True)

    for category in draw_order:
        color = COLOR_MAPPING.get(category, '#cccccc')
        x = plot_df['Zeit']
        val_orig = plot_df[category].values
        total = plot_df['Total_Month'].values
        
        if mode == 'stacked':
            # Additiver Plot: Obere Kante berechnen
            y_top = current_stack + val_orig
            
            plt.fill_between(x, y_top, current_stack, color=color, alpha=0.8)
            plt.plot(x, y_top, marker='o', color=color, label=category, linewidth=2, markersize=6)
            
            # Prozent-Labels an der Oberkante des Segments
            for i, val in enumerate(val_orig):
                if total[i] > 0:
                    percent = (val / total[i]) * 100
                    plt.text(i, y_top[i] + (total.max() * 0.015), f'{percent:.1f}%', 
                             ha='center', va='bottom', fontsize=9, color=color, fontweight='bold')
            
            current_stack = y_top # Basis für den nächsten Stapel erhöhen
        else:
            # Klassischer überlappender Plot
            plt.fill_between(x, val_orig, 0, color=color, alpha=0.6)
            plt.plot(x, val_orig, marker='o', color=color, label=category, linewidth=2, markersize=6)
            
            for i, val in enumerate(val_orig):
                if total[i] > 0:
                    percent = (val / total[i]) * 100
                    plt.text(i, val + (total.max() * 0.015), f'{percent:.1f}%', 
                             ha='center', va='bottom', fontsize=9, color=color, fontweight='bold')

    # Layout & Styling
    plt.title(f'Pkw-Neuzulassungen: {country_name} ({mode.capitalize()})', fontsize=18, pad=35)
    plt.xlabel('Berichtsmonat', fontsize=12)
    plt.ylabel('Anzahl der Neuzulassungen', fontsize=12)
    plt.xticks(range(len(plot_df)), plot_df['Zeit'], rotation=45)
    plt.grid(True, linestyle='--', alpha=0.3, zorder=0)
    
    # Legende in fester Reihenfolge
    handles, labels = plt.gca().get_legend_handles_labels()
    handle_dict = dict(zip(labels, handles))
    legend_order = [c for c in CATEGORY_MAPPING.keys() if c in handle_dict]
    sorted_handles = [handle_dict[l] for l in legend_order]
    
    plt.legend(sorted_handles, legend_order, title="Fahrzeugklasse", 
               bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    
    # Speichern
    file_safe_name = country_name.replace(" ", "_").lower()
    output_filename = f"plot_{mode}_{file_safe_name}.png"
    plt.savefig(output_filename)
    print(f"Diagramm im Modus '{mode}' gespeichert unter: {output_filename}")


# --- HAUPTPROGRAMM ---
if __name__ == "__main__":
    # Pfad zu deiner Datei
    file_to_open = r"C:\Users\marce\OneDrive\HS-Kempten\Wintersemester_1\Grundlagen_Fahrerassistenz\202511_NZL_Pkw_KREN_csv_Deutschland.csv"
    
    # Ausführung
    process_and_plot(file_to_open, "Deutschland", PLOT_MODE)

    # Pfad zu deiner Datei
    file_to_open = r"C:\Users\marce\OneDrive\HS-Kempten\Wintersemester_1\Grundlagen_Fahrerassistenz\202511_NZL_Pkw_KREN_csv_China.csv"
    
    # Ausführung
    process_and_plot(file_to_open, "China", PLOT_MODE)