import os
import glob
import pandas as pd

def convert_trc_to_jmp(filepath):
    """Special parser to clean OpenSim .trc headers and create a flat .csv for JMP"""
    with open(filepath, 'r') as f:
        lines = f.readlines()
        
    start_row = 0
    header_line = ""
    # Find the header row starting with Frame# or Time
    for i in range(min(20, len(lines))):
        if 'Frame#' in lines[i] or 'Time' in lines[i] or 'time' in lines[i].lower() or 'frame#' in lines[i].lower():
            start_row = i
            header_line = lines[i]
            break
            
    if start_row == 0:
        print(f"❌ Could not find header row in {os.path.basename(filepath)}")
        return
        
    # Read the main column markers
    main_cols = [c.strip() for c in header_line.split('\t') if c.strip() != '']
    data_start = start_row + 2
    
    try:
        df = pd.read_csv(filepath, sep='\t', skiprows=data_start, header=None)
        
        flat_cols = []
        flat_cols.append("Frame")
        flat_cols.append("Time")
        
        marker_idx = 2
        for col_idx in range(2, len(df.columns)):
            if marker_idx < len(main_cols):
                marker_name = main_cols[marker_idx]
            else:
                marker_name = f"M{marker_idx}"
                
            axis_idx = (col_idx - 2) % 3
            if axis_idx == 0: axis = 'X'
            elif axis_idx == 1: axis = 'Y'
            else: 
                axis = 'Z'
                marker_idx += 1 
                
            flat_cols.append(f"{marker_name}_{axis}")
            
        df.columns = flat_cols[:len(df.columns)]
        
        out_path = filepath.replace('.trc', '_JMP.csv')
        df.to_csv(out_path, index=False)
        print(f"✅ Converted: {os.path.basename(filepath)} -> {os.path.basename(out_path)}")
    except Exception as e:
        print(f"❌ Error converting {os.path.basename(filepath)}: {e}")

def convert_mot_to_jmp(filepath):
    """Special parser to clean OpenSim .mot headers and create a flat .csv for JMP"""
    with open(filepath, 'r') as f:
        lines = f.readlines()
        
    start_row = 0
    for i in range(min(50, len(lines))):
        if 'time' in lines[i].lower():
            start_row = i
            break
            
    try:
        df = pd.read_csv(filepath, sep='\t', skiprows=start_row)
        df.columns = df.columns.str.strip()
        out_path = filepath.replace('.mot', '_JMP.csv')
        df.to_csv(out_path, index=False)
        print(f"✅ Converted: {os.path.basename(filepath)} -> {os.path.basename(out_path)}")
    except Exception as e:
        print(f"❌ Error converting {os.path.basename(filepath)}: {e}")

def main():
    print("="*50)
    print("OpenSim to JMP CSV Converter (Scanning Downloads & Desktop)")
    print("="*50)
    
    downloads_path = os.path.expanduser("~/Downloads")
    desktop_path = os.path.expanduser("~/Desktop")
    
    # Search recursively in both Downloads and Desktop folders
    trc_files = glob.glob(os.path.join(downloads_path, "**/*.trc"), recursive=True) + \
                glob.glob(os.path.join(desktop_path, "**/*.trc"), recursive=True)
                
    mot_files = glob.glob(os.path.join(downloads_path, "**/*.mot"), recursive=True) + \
                glob.glob(os.path.join(desktop_path, "**/*.mot"), recursive=True)
    
    if not trc_files and not mot_files:
        print("No .trc or .mot files found anywhere!")
        return
        
    for f in trc_files:
        # Avoid converting already converted files if some weird naming happens
        if '_JMP.csv' not in f:
            convert_trc_to_jmp(f)
        
    for f in mot_files:
        if '_JMP.csv' not in f:
            convert_mot_to_jmp(f)
        
    print(f"\n✅ All {len(trc_files)} TRC files and {len(mot_files)} MOT files processed!")

if __name__ == '__main__':
    main()
