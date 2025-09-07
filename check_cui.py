# check_cui.py

# check_cui.py

import pandas as pd
import os

# --- Configuration: File Paths ---
# Input file in the project root whose CUIs need to be checked
# MODIFIED: Changed from 'submission_to_check.csv' to 'submission.csv'
submission_to_check_file = 'submission.csv'

# Validation CUI list file in ./data/test/
# This file should have a column named 'CUI' containing all valid CUI codes.
data_test_dir = os.path.join('.', 'data', 'test')
cui_names_file = os.path.join(data_test_dir, 'cui_names.csv')

# Output file in the project root (will overwrite if it exists)
# This is the desired final output file name.
submission_output_file = 'submission_cui.csv'

# --- Helper function to create dummy files for testing (optional) ---
def create_dummy_files_for_development_checker():
    """
    Creates dummy input files for testing the CUI checking script.
    This is useful if you want to test the script's logic without your actual files.
    Ensure the ./data/test/ directory can be created or already exists.
    """
    print("Creating dummy files for CUI checker development testing...")
    os.makedirs(data_test_dir, exist_ok=True) # Create ./data/test if it doesn't exist

    # Dummy submission.csv (ID, CUIs with mixed validity)
    # MODIFIED: Changed dummy file name to submission.csv for consistency
    data_submission_to_check = {
        'ID': ['S_ID_001', 'S_ID_002', 'S_ID_003', 'S_ID_004', 'S_ID_005', 'S_ID_006', 'S_ID_007'],
        'CUIs': [
            'C00VALID1;C00INVALID1;C00VALID2', # Mix of valid and invalid
            'C00VALID3',                       # Single valid CUI
            'C00INVALID2;C00INVALID3',         # All CUIs are invalid
            '',                                # Empty string for CUIs
            None,                              # Represents a NaN or missing CUI field
            'C00VALID1; C00VALID2 ',           # Valid CUIs with extra spaces
            'C00UNKNOWN'                       # Single CUI not in cui_names
        ]
    }
    df_submission_to_check = pd.DataFrame(data_submission_to_check)
    df_submission_to_check.to_csv(submission_to_check_file, index=False) # Uses the modified submission_to_check_file variable
    print(f"Created dummy '{submission_to_check_file}'")

    # Dummy cui_names.csv (source of valid CUIs)
    # Must contain a 'CUI' column. Other columns like 'Name' are ignored for validation.
    data_cui_names_validator = {
        'CUI': ['C00VALID1', 'C00VALID2', 'C00VALID3', 'C00VALID4'],
        'Name': ['Valid Concept Name 1', 'Valid Concept Name 2', 'Valid Concept Name 3', 'Valid Concept Name 4']
    }
    df_cui_names_validator = pd.DataFrame(data_cui_names_validator)
    df_cui_names_validator.to_csv(cui_names_file, index=False)
    print(f"Created dummy '{cui_names_file}' (for CUI checker script)")
    print("-" * 30)

# --- Main CUI Filtering Logic ---
def filter_invalid_cuis():
    """
    Reads the input submission CSV and validates its CUIs against 'cui_names.csv'.
    Removes CUIs not found in 'cui_names.csv'.
    Saves the cleaned data, overwriting the input 'submission.csv'.
    """
    print(f"Starting CUI filtering process for '{submission_to_check_file}' based on presence in '{cui_names_file}'...")
    try:
        # Load the submission file that needs its CUIs checked
        df_submission_to_check = pd.read_csv(submission_to_check_file)
        print(f"Successfully loaded '{submission_to_check_file}' ({len(df_submission_to_check)} rows).")

        # Load the file containing the list of valid CUIs
        df_cui_names = pd.read_csv(cui_names_file)
        print(f"Successfully loaded '{cui_names_file}' for validation.")

    except FileNotFoundError as e:
        print(f"Error: Could not read one or more input files. Details: {e}")
        print(f"Please ensure '{submission_to_check_file}' is in the project root.")
        print(f"And ensure '{cui_names_file}' (with a 'CUI' column) is in '{data_test_dir}'.")
        return False
    except Exception as e:
        print(f"An unexpected error occurred while loading files: {e}")
        return False

    # Check for the presence of essential columns
    if 'CUI' not in df_cui_names.columns:
        print(f"Error: 'CUI' column not found in '{cui_names_file}'. This column is required for validation.")
        return False
    
    if 'CUIs' not in df_submission_to_check.columns:
        print(f"Error: 'CUIs' column not found in '{submission_to_check_file}'. No CUIs to filter.")
        return False
    
    valid_cuis_set = set(df_cui_names['CUI'].astype(str).unique())
    if not valid_cuis_set:
        print(f"Warning: No valid CUIs found in '{cui_names_file}' or 'CUI' column is empty. All CUIs in '{submission_to_check_file}' will be considered invalid and erased.")
    else:
        print(f"Loaded {len(valid_cuis_set)} unique valid CUI codes from '{cui_names_file}'.")


    def filter_and_rejoin_cuis(cui_string_to_filter):
        if pd.isna(cui_string_to_filter) or str(cui_string_to_filter).strip() == "":
            return "" 
        
        individual_cuis = str(cui_string_to_filter).split(';')
        kept_cuis = [] 
        
        for cui in individual_cuis:
            cleaned_cui = cui.strip() 
            if cleaned_cui and cleaned_cui in valid_cuis_set:
                kept_cuis.append(cleaned_cui)
        
        return ";".join(kept_cuis)

    # Apply the filtering function to the 'CUIs' column.
    # The original 'CUIs' column is modified directly or a new one is created and then renamed.
    # Here, we create a new column and then decide how to handle it for the output.
    df_submission_to_check['Filtered_CUIs'] = df_submission_to_check['CUIs'].apply(filter_and_rejoin_cuis)
    
    # For the output, we want the original columns, but with 'CUIs' updated.
    # If an 'ID' column exists, keep it.
    output_columns = []
    if 'ID' in df_submission_to_check.columns:
        output_columns.append('ID')
    
    # Replace the old 'CUIs' column with the new 'Filtered_CUIs' content
    df_submission_to_check['CUIs'] = df_submission_to_check['Filtered_CUIs']
    output_columns.append('CUIs')

    # Include any other columns from the original file
    for col in df_submission_to_check.columns:
        if col not in output_columns and col != 'Filtered_CUIs':
            output_columns.append(col)
            
    df_output_submission = df_submission_to_check[output_columns]

    try:
        df_output_submission.to_csv(submission_output_file, index=False)
        print(f"\nFiltering process completed successfully.")
        print(f"Output saved to '{submission_output_file}', overwriting the original file with updated CUI lists.")

        original_cui_counts = df_submission_to_check['CUIs'].fillna('').astype(str).apply(lambda x: len(x.split(';')) if x else 0) # Count based on potentially just filtered CUIs if column was overwritten early
        # To get a true original count, we might need to re-read or preserve the original CUIs column before modification if detailed stats are critical.
        # For this adaptation, the current logic for stats should reflect changes based on the 'Filtered_CUIs' column.
        
        # Recalculate counts for stats based on the final output dataframe
        final_cui_counts = df_output_submission['CUIs'].fillna('').astype(str).apply(lambda x: len(x.split(';')) if x else 0)
  
        # Reconstructing output_df as per original script's intent for stats:
        temp_df_for_output = pd.DataFrame()
        if 'ID' in df_submission_to_check.columns:
            temp_df_for_output['ID'] = df_submission_to_check['ID']
        temp_df_for_output['CUIs'] = df_submission_to_check['Filtered_CUIs'] # This column has the filtered CUIs
        
        # Add other original columns back
        for col in df_submission_to_check.columns:
            if col not in ['ID', 'CUIs', 'Filtered_CUIs']: # Avoid duplicating ID/CUIs, and don't add helper
                 temp_df_for_output[col] = df_submission_to_check[col]
        
        # Ensure 'CUIs' is the name of the column with filtered data in the output
        # If 'CUIs' was not among other original columns, it's fine. If it was, it's now correctly from Filtered_CUIs.
        
        df_output_final = temp_df_for_output.copy()
        df_output_final.to_csv(submission_output_file, index=False)
        
        # Stats calculation based on original 'CUIs' in df_submission_to_check and new 'CUIs' in df_output_final
        original_cui_strings_for_stats = df_submission_to_check['CUIs'].fillna('').astype(str)
        filtered_cui_strings_for_stats = df_output_final['CUIs'].fillna('').astype(str)

        original_cui_counts_stat = original_cui_strings_for_stats.apply(lambda x: len(x.split(';')) if x and x.strip() else 0)
        filtered_cui_counts_stat = filtered_cui_strings_for_stats.apply(lambda x: len(x.split(';')) if x and x.strip() else 0)
        
        modified_rows_count = (original_cui_counts_stat != filtered_cui_counts_stat).sum()
        # A row is also modified if it went from "C1;C2" to "" (empty string)
        # The condition should ideally be: original_string != filtered_string
        # Let's use direct string comparison for modification status for robustness
        modified_rows_count_alt = (original_cui_strings_for_stats != filtered_cui_strings_for_stats).sum()
        
        print(f"Number of rows where CUIs list was modified: {modified_rows_count_alt} out of {len(df_submission_to_check)}.")
        # Optional: More detailed stats
        total_original_cuis = original_cui_counts_stat.sum()
        total_filtered_cuis = filtered_cui_counts_stat.sum()
        print(f"Total CUIs before filtering (approx): {total_original_cuis}")
        print(f"Total CUIs after filtering: {total_filtered_cuis}")
        print(f"{total_original_cuis - total_filtered_cuis} CUIs were removed.")
        return True
    except IOError as e:
        print(f"Error: Could not write the output file '{submission_output_file}'. Details: {e}")
        return False
    except Exception as e:
        print(f"An unexpected error occurred while saving the output file: {e}")
        return False

# --- Main execution logic ---
if __name__ == '__main__':
    # --- Optional: Create dummy files for a quick test run ---
    # create_dummy_files_for_development_checker() # Make sure this uses the new 'submission.csv' as input for dummy
    # --- End of optional dummy file creation ---

    print("CUI Filtering Script Initializing...")
    print("-" * 50)
    
    success = filter_invalid_cuis()
    
    print("-" * 50)
    if success:
        print("Script finished successfully.")
    else:
        print("Script encountered errors. Please review the messages above.")
        
        