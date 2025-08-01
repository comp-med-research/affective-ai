#!/usr/bin/env python3
"""
Script to create a stratified sample of 100 examples from test_merged.csv
for quick inference testing across all emotion categories.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse

def get_stratified_sample(input_file: str, output_file: str, sample_size: int = 100, random_state: int = 42):
    """
    Create a stratified sample from the input CSV file based on the Emotion column.
    
    Args:
        input_file: Path to the input CSV file
        output_file: Path to save the stratified sample
        sample_size: Total number of samples to extract (default: 100)
        random_state: Random seed for reproducibility (default: 42)
    """
    
    # Load the data
    print(f"📊 Loading data from {input_file}...")
    df = pd.read_csv(input_file)
    
    print(f"   Total samples: {len(df)}")
    print(f"   Columns: {list(df.columns)}")
    
    # Check emotion distribution
    emotion_counts = df['Emotion'].value_counts()
    print(f"\n🎭 Emotion distribution in source data:")
    for emotion, count in emotion_counts.items():
        print(f"   {emotion}: {count} ({count/len(df)*100:.1f}%)")
    
    # Calculate samples per emotion (proportional to original distribution)
    emotion_proportions = df['Emotion'].value_counts(normalize=True)
    samples_per_emotion = (emotion_proportions * sample_size).round().astype(int)
    
    # Adjust if rounding causes total to not equal sample_size
    total_samples = samples_per_emotion.sum()
    if total_samples != sample_size:
        # Add/remove samples from the most frequent emotion
        most_frequent = emotion_proportions.index[0]
        adjustment = sample_size - total_samples
        samples_per_emotion[most_frequent] += adjustment
    
    print(f"\n🎯 Target stratified sample (n={sample_size}):")
    for emotion, count in samples_per_emotion.items():
        print(f"   {emotion}: {count} samples")
    
    # Create stratified sample
    sampled_dfs = []
    np.random.seed(random_state)
    
    for emotion, target_count in samples_per_emotion.items():
        emotion_data = df[df['Emotion'] == emotion]
        
        if len(emotion_data) < target_count:
            print(f"⚠️  Warning: Only {len(emotion_data)} samples available for {emotion}, requesting {target_count}")
            sampled = emotion_data
        else:
            sampled = emotion_data.sample(n=target_count, random_state=random_state)
        
        sampled_dfs.append(sampled)
    
    # Combine all samples
    stratified_sample = pd.concat(sampled_dfs, ignore_index=True)
    
    # Shuffle the final dataset
    stratified_sample = stratified_sample.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    # Save the result
    print(f"\n💾 Saving stratified sample to {output_file}...")
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stratified_sample.to_csv(output_file, index=False)
    
    # Verify the final distribution
    final_emotion_counts = stratified_sample['Emotion'].value_counts()
    print(f"\n✅ Final sample distribution:")
    for emotion, count in final_emotion_counts.items():
        print(f"   {emotion}: {count} ({count/len(stratified_sample)*100:.1f}%)")
    
    print(f"\n🎉 Successfully created inference set with {len(stratified_sample)} samples!")
    return stratified_sample

def main():
    parser = argparse.ArgumentParser(description="Create stratified inference set from test data")
    parser.add_argument(
        "--input", 
        default="data/merged/test_merged.csv",
        help="Input CSV file path (default: data/merged/test_merged.csv)"
    )
    parser.add_argument(
        "--output",
        default="data/merged/inference_set.csv", 
        help="Output CSV file path (default: data/merged/inference_set.csv)"
    )
    parser.add_argument(
        "--size",
        type=int,
        default=100,
        help="Number of samples in inference set (default: 100)"
    )
    parser.add_argument(
        "--seed",
        type=int, 
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not Path(args.input).exists():
        print(f"❌ Error: Input file {args.input} not found!")
        return 1
    
    try:
        get_stratified_sample(args.input, args.output, args.size, args.seed)
        return 0
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main()) 