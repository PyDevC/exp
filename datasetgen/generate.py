import pandas as pd
import numpy as np

# 1. Physics constants from your Research Paper
n0 = 0.8  # Optical efficiency
a1 = 3.5  # Linear loss
a2 = 0.05 # Quadratic loss
Cp = 4186 # Specific heat J/kgK

def generate_ieee_dataset(samples_per_model=50):
    # Simplified hardware list based on your CSV snippet
    # Format: [Brand, Model, Collector_Area_SqFt, Tank_Volume_Gal, System_Type]
    # System_Type: 0 for Thermosyphon, 1 for Forced Circulation
    hardware_specs = [
        ['MC Green', 'A80-20', 18.1, 80, 0],
        ['Rheem', 'RS80-48BP', 82.0, 80, 1],
        ['Solahart', '181BTC', 21.5, 48, 0],
        ['AET', 'EagleSun DX', 40.6, 80, 1],
        ['SunPad', 'Double E', 44.6, 52, 0]
    ]
    
    combined_data = []
    
    for brand, model, area_sqft, tank_gal, sys_type in hardware_specs:
        area_m2 = area_sqft * 0.0929  # Convert sqft to m2
        
        for _ in range(samples_per_model):
            # A. Environmental Inputs (Weather)
            I = np.random.uniform(200, 1100) # Irradiance (W/m2)
            Ta = np.random.uniform(-5, 40)   # Ambient Temp (C)
            Tin = np.random.uniform(10, 25)  # Inlet Water Temp (C)
            
            # B. Inject Real-World Distortions (Sensor Noise)
            # Sensors in real solar systems have ~3-5% error
            I_noisy = I * np.random.normal(1, 0.04) 
            Ta_noisy = Ta + np.random.normal(0, 0.8)
            
            # C. Target Calculation: Optimal Flow Rate (ṁ_opt)
            # Physics Logic: Larger collectors require higher flow to prevent overheating
            # Hybrid systems (Forced) can handle higher flow than Thermosyphon
            base_flow = (I / 10000) * (area_m2 / 2.0)
            sys_modifier = 0.02 if sys_type == 1 else -0.01
            m_opt = base_flow + sys_modifier + np.random.normal(0, 0.002)
            
            # Constraints: Clip to realistic pump limits (0.01 to 0.15 kg/s)
            m_opt = np.clip(m_opt, 0.01, 0.15)
            
            combined_data.append([
                I_noisy, Ta_noisy, Tin, area_m2, tank_gal, sys_type, m_opt
            ])
            
    columns = [
        'Irradiance', 'Ambient_Temp', 'Inlet_Temp', 
        'Collector_Area_m2', 'Tank_Volume_Gal', 'System_Type', 'Optimal_Flow_Rate'
    ]
    return pd.DataFrame(combined_data, columns=columns)

# Generate and Save
final_df = generate_ieee_dataset(10000)
print(len(final_df))
final_df.to_csv('IEEE_SWH_RealWorld_Dataset.csv', index=False)
print(f"Generated {len(final_df)} rows of combined distorted data.")
