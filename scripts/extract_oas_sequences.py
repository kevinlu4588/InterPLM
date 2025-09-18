#!/usr/bin/env python3
"""
Extract paired amino acid sequences from Observed Antibody Space (OAS) CSV files
and save as FASTA format with sequence counts.
"""

import pandas as pd
import gzip
import os
import glob
from pathlib import Path
import argparse
from collections import Counter


def extract_aa_sequence_from_full_sequence(full_sequence, chain_type='heavy'):
    """
    Extract amino acid sequence from the sequence_alignment_aa column.
    This contains the full translated sequence including framework and CDR regions.
    """
    if pd.isna(full_sequence) or full_sequence == '':
        return None
    
    # Remove any 'X' characters which represent unknown amino acids
    # Keep the sequence as is - it already contains the variable region
    clean_sequence = str(full_sequence).replace('X', '')
    
    # Basic validation - should be reasonable length for antibody variable region
    if len(clean_sequence) < 50:  # Too short for a variable region
        return None
    
    return clean_sequence


def process_csv_file(file_path, output_fasta, sequence_counter, file_counter):
    """Process a single CSV file and extract paired sequences."""
    
    # Convert PosixPath to string if needed
    file_path_str = str(file_path)
    print(f"Processing: {file_path_str}")
    
    try:
        # Handle both gzipped and regular CSV files
        if file_path_str.endswith('.gz'):
            df = pd.read_csv(file_path_str, compression='gzip', sep='\t', low_memory=False)
        else:
            df = pd.read_csv(file_path_str, sep='\t', low_memory=False)
        
        if df.empty:
            print(f"  Empty file: {file_path_str}")
            return 0
        
        print(f"  Found {len(df)} rows")
        
        # Check if this file has paired data (both heavy and light chains)
        has_heavy = any(col.endswith('_heavy') for col in df.columns)
        has_light = any(col.endswith('_light') for col in df.columns)
        
        sequences_added = 0
        
        if has_heavy and has_light:
            # Process paired sequences
            for idx, row in df.iterrows():
                # Extract heavy chain AA sequence
                heavy_aa = None
                if 'sequence_alignment_aa_heavy' in df.columns:
                    heavy_aa = extract_aa_sequence_from_full_sequence(row['sequence_alignment_aa_heavy'], 'heavy')
                
                # Extract light chain AA sequence  
                light_aa = None
                if 'sequence_alignment_aa_light' in df.columns:
                    light_aa = extract_aa_sequence_from_full_sequence(row['sequence_alignment_aa_light'], 'light')
                
                # Only include if we have both heavy and light sequences
                if heavy_aa and light_aa:
                    sequence_id = row.get('sequence_id_heavy', f"seq_{file_counter}_{sequences_added}")
                    
                    # Write heavy chain
                    output_fasta.write(f">{sequence_id}_heavy\n{heavy_aa}\n")
                    
                    # Write light chain  
                    output_fasta.write(f">{sequence_id}_light\n{light_aa}\n")
                    
                    sequences_added += 1
                    sequence_counter['paired'] += 1
                    
        elif has_heavy:
            # Process heavy chain only
            for idx, row in df.iterrows():
                heavy_aa = None
                if 'sequence_alignment_aa_heavy' in df.columns:
                    heavy_aa = extract_aa_sequence_from_full_sequence(row['sequence_alignment_aa_heavy'], 'heavy')
                
                if heavy_aa:
                    sequence_id = row.get('sequence_id_heavy', f"seq_{file_counter}_{sequences_added}")
                    output_fasta.write(f">{sequence_id}_heavy\n{heavy_aa}\n")
                    sequences_added += 1
                    sequence_counter['heavy_only'] += 1
        
        elif has_light:
            # Process light chain only
            for idx, row in df.iterrows():
                light_aa = None
                if 'sequence_alignment_aa_light' in df.columns:
                    light_aa = extract_aa_sequence_from_full_sequence(row['sequence_alignment_aa_light'], 'light')
                
                if light_aa:
                    sequence_id = row.get('sequence_id_light', f"seq_{file_counter}_{sequences_added}")
                    output_fasta.write(f">{sequence_id}_light\n{light_aa}\n")
                    sequences_added += 1
                    sequence_counter['light_only'] += 1
        
        print(f"  Extracted {sequences_added} sequences")
        return sequences_added
        
    except Exception as e:
        print(f"  Error processing {file_path_str}: {str(e)}")
        return 0


def main():
    parser = argparse.ArgumentParser(description='Extract amino acid sequences from OAS CSV files')
    parser.add_argument('input_dir', help='Directory containing CSV.gz files from OAS')
    parser.add_argument('-o', '--output', default='oas_sequences.fasta', 
                       help='Output FASTA file (default: oas_sequences.fasta)')
    parser.add_argument('-p', '--pattern', default='*.csv.gz',
                       help='File pattern to match (default: *.csv.gz)')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_file = args.output
    
    if not input_dir.exists():
        print(f"Error: Input directory {input_dir} does not exist")
        return
    
    # Find all CSV files
    csv_files = list(input_dir.glob(args.pattern))
    
    # Also look for uncompressed CSV files
    if args.pattern == '*.csv.gz':
        csv_files.extend(list(input_dir.glob('*.csv')))
    
    if not csv_files:
        print(f"No CSV files found in {input_dir} matching pattern {args.pattern}")
        return
    
    print(f"Found {len(csv_files)} CSV files to process")
    
    # Counters
    sequence_counter = Counter()
    total_sequences = 0
    file_counter = 0
    
    # Process all files
    with open(output_file, 'w') as fasta_out:
        for csv_file in csv_files:
            file_counter += 1
            sequences_in_file = process_csv_file(csv_file, fasta_out, sequence_counter, file_counter)
            total_sequences += sequences_in_file
    
    # Print summary
    print(f"\n=== SUMMARY ===")
    print(f"Files processed: {len(csv_files)}")
    print(f"Total sequences extracted: {total_sequences}")
    print(f"  - Paired sequences: {sequence_counter['paired']} pairs ({sequence_counter['paired']*2} total chains)")
    print(f"  - Heavy chain only: {sequence_counter['heavy_only']}")
    print(f"  - Light chain only: {sequence_counter['light_only']}")
    print(f"Output written to: {output_file}")


if __name__ == "__main__":
    main()