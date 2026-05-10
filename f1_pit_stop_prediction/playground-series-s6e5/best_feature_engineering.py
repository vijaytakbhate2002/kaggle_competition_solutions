def engineer_race_features(df):
    """
    Applies feature engineering for race strategy prediction.
    Handles high correlation via ratios and extracts stint-based metrics.
    """
    # Create a copy to avoid SettingWithCopyWarning
    df = df.copy()

    # 1. Handling your high correlation pair (RaceProgress / LapNumber)
    # Adding a small epsilon to avoid division by zero
    df['Progress_Per_Lap_engg'] = df['RaceProgress'] / (df['LapNumber'] + 1e-5)

    # 2. Tyre & Degradation Ratios
    # Captures the intensity of degradation relative to the distance traveled
    df['Deg_Per_Lap_engg'] = df['Cumulative_Degradation'] / (df['LapNumber'] + 1e-5)
    df['Deg_Per_TyreLife_engg'] = df['Cumulative_Degradation'] / (df['TyreLife'] + 1e-5)

    # 4. Pace Sensitivity (The "Cliff" Detector)
    # How much is the lap time changing relative to tyre age?
    df['Pace_Tyre_Sensitivity_engg'] = df['LapTime_Delta'] / (df['TyreLife'] + 1e-5)

    # 5. Strategic Flags
    # Identify if a driver is losing positions (potential pressure to pit)
    df['Losing_Ground_engg'] = (df['Position_Change'] < 0).astype(int)

    # Identify 'Fresh' vs 'Old' tyres based on stint start
    df['Is_Late_Stint_engg'] = (df['TyreLife'] > 20).astype(int)

    # 6. Interaction Terms
    # Multiplying LapTime_Delta by Cumulative_Degradation to highlight
    # laps where both pace drops and wear is high
    df['Wear_Pace_Impact_engg'] = df['LapTime_Delta'] * df['Cumulative_Degradation']

    return df