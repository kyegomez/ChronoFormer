"""
Time Series Data Preprocessor
----------------------------
Handles loading, preprocessing, and transformation of time series data from various sources.
Includes support for CSV, Excel, JSON, and pandas DataFrame inputs.

Features:
- Automatic data type detection and parsing
- Missing value handling
- Feature scaling and normalization
- Sliding window creation
- Time-based feature engineering
- Memory-efficient data loading for large datasets
"""

import pandas as pd
import numpy as np
import torch
from typing import Union, Optional, List, Dict, Tuple
from pathlib import Path
from datetime import datetime
from loguru import logger
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pyarrow as pa
import pyarrow.parquet as pq
from dateutil.parser import parse
import json

class TimeSeriesPreprocessor:
    """Preprocesses time series data for the efficient transformer model."""
    
    def __init__(
        self,
        time_column: str,
        feature_columns: List[str],
        target_columns: Optional[List[str]] = None,
        sequence_length: int = 100,
        stride: int = 1,
        batch_size: int = 32,
        scaling_method: str = 'standard',
        fill_method: str = 'forward',
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize the preprocessor.
        
        Args:
            time_column: Name of the timestamp column
            feature_columns: List of feature column names
            target_columns: List of target column names (if different from features)
            sequence_length: Length of sequences to generate
            stride: Stride for sliding window
            batch_size: Batch size for data loading
            scaling_method: 'standard' or 'minmax'
            fill_method: Method for handling missing values
            device: Device to store tensors on
        """
        self.time_column = time_column
        self.feature_columns = feature_columns
        self.target_columns = target_columns or feature_columns
        self.sequence_length = sequence_length
        self.stride = stride
        self.batch_size = batch_size
        self.scaling_method = scaling_method
        self.fill_method = fill_method
        self.device = device
        
        # Initialize scalers
        self.scalers = {}
        for col in self.feature_columns:
            self.scalers[col] = (
                StandardScaler() if scaling_method == 'standard' else MinMaxScaler()
            )
            
        logger.info(f"Initialized TimeSeriesPreprocessor with {len(feature_columns)} features")
    
    def _validate_file(self, file_path: Union[str, Path]) -> Path:
        """Validate file existence and format."""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        supported_extensions = {'.csv', '.xlsx', '.parquet', '.json'}
        if file_path.suffix not in supported_extensions:
            raise ValueError(f"Unsupported file format. Supported formats: {supported_extensions}")
        
        return file_path
    
    def _parse_time(self, time_series: pd.Series) -> pd.Series:
        """Parse time column to datetime."""
        try:
            if time_series.dtype == 'object':
                return pd.to_datetime(time_series.apply(parse))
            return pd.to_datetime(time_series)
        except Exception as e:
            logger.error(f"Error parsing time column: {str(e)}")
            raise
    
    def load_data(
        self,
        source: Union[str, Path, pd.DataFrame],
        chunk_size: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Load data from various sources.
        
        Args:
            source: File path or DataFrame
            chunk_size: Size of chunks for large files
            
        Returns:
            Loaded DataFrame
        """
        try:
            if isinstance(source, pd.DataFrame):
                df = source
            else:
                file_path = self._validate_file(source)
                
                if chunk_size:
                    # Memory-efficient loading for large files
                    if file_path.suffix == '.parquet':
                        return pq.read_table(file_path).to_pandas()
                    elif file_path.suffix == '.csv':
                        return pd.read_csv(file_path, chunksize=chunk_size)
                    elif file_path.suffix == '.xlsx':
                        return pd.read_excel(file_path, chunksize=chunk_size)
                    else:  # JSON
                        with open(file_path) as f:
                            return pd.DataFrame(json.load(f))
                else:
                    # Regular loading for smaller files
                    if file_path.suffix == '.parquet':
                        df = pq.read_table(file_path).to_pandas()
                    elif file_path.suffix == '.csv':
                        df = pd.read_csv(file_path)
                    elif file_path.suffix == '.xlsx':
                        df = pd.read_excel(file_path)
                    else:  # JSON
                        with open(file_path) as f:
                            df = pd.DataFrame(json.load(f))
            
            # Validate columns
            missing_cols = (
                set(self.feature_columns + [self.time_column]) - set(df.columns)
            )
            if missing_cols:
                raise ValueError(f"Missing columns in data: {missing_cols}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def create_sequences(
        self,
        data: pd.DataFrame
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Create sequences for training using sliding windows.
        
        Args:
            data: Preprocessed DataFrame
            
        Returns:
            Tuple of (features, timestamps, targets)
        """
        sequences = []
        timestamps = []
        targets = []
        
        # Convert timestamps to indices
        time_indices = pd.factorize(data[self.time_column])[0]
        
        # Create sliding windows
        for i in range(0, len(data) - self.sequence_length + 1, self.stride):
            # Extract sequence
            sequence = data[self.feature_columns].iloc[i:i + self.sequence_length].values
            target = data[self.target_columns].iloc[i + self.sequence_length - 1].values
            time_idx = time_indices[i:i + self.sequence_length]
            
            sequences.append(sequence)
            timestamps.append(time_idx)
            targets.append(target)
        
        # Convert to tensors
        return (
            torch.FloatTensor(sequences).to(self.device),
            torch.LongTensor(timestamps).to(self.device),
            torch.FloatTensor(targets).to(self.device)
        )
    
    def preprocess(
        self,
        source: Union[str, Path, pd.DataFrame],
        fit_scalers: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Main preprocessing pipeline.
        
        Args:
            source: Data source
            fit_scalers: Whether to fit or just transform with scalers
            
        Returns:
            Preprocessed sequences ready for model
        """
        logger.info("Starting preprocessing pipeline")
        
        try:
            # Load data
            df = self.load_data(source)
            
            # Parse time column
            df[self.time_column] = self._parse_time(df[self.time_column])
            
            # Sort by time
            df = df.sort_values(self.time_column)
            
            # Handle missing values
            df[self.feature_columns] = df[self.feature_columns].fillna(
                method=self.fill_method
            )
            
            # Scale features
            scaled_features = df[self.feature_columns].copy()
            for col in self.feature_columns:
                if fit_scalers:
                    scaled_features[col] = self.scalers[col].fit_transform(
                        df[[col]]
                    )
                else:
                    scaled_features[col] = self.scalers[col].transform(
                        df[[col]]
                    )
            
            df[self.feature_columns] = scaled_features
            
            # Create sequences
            sequences = self.create_sequences(df)
            
            logger.info(
                f"Preprocessing complete. Created {len(sequences[0])} sequences"
            )
            
            return sequences
            
        except Exception as e:
            logger.error(f"Error in preprocessing pipeline: {str(e)}")
            raise
    
    def create_dataloader(
        self,
        sequences: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        shuffle: bool = True
    ) -> torch.utils.data.DataLoader:
        """Create DataLoader from sequences."""
        dataset = torch.utils.data.TensorDataset(*sequences)
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle
        )
    
    def inverse_transform(
        self,
        scaled_data: torch.Tensor,
        columns: List[str]
    ) -> torch.Tensor:
        """
        Inverse transform scaled data back to original scale.
        
        Args:
            scaled_data: Scaled tensor
            columns: Column names corresponding to tensor dimensions
            
        Returns:
            Tensor in original scale
        """
        # Convert to numpy for inverse transform
        data_np = scaled_data.cpu().numpy()
        
        # Inverse transform each column
        for i, col in enumerate(columns):
            if col in self.scalers:
                data_np[..., i] = self.scalers[col].inverse_transform(
                    data_np[..., i].reshape(-1, 1)
                ).ravel()
        
        return torch.FloatTensor(data_np).to(self.device)

# Example usage
if __name__ == "__main__":
    # Example configuration
    config = {
        "time_column": "timestamp",
        "feature_columns": ["temperature", "humidity", "pressure"],
        "sequence_length": 24,  # 24 time steps sequence
        "stride": 1,
        "batch_size": 32
    }
    
    try:
        # Initialize preprocessor
        preprocessor = TimeSeriesPreprocessor(**config)
        
        # Example with CSV file
        sequences = preprocessor.preprocess("sensor_data.csv")
        
        # Create DataLoader
        dataloader = preprocessor.create_dataloader(sequences)
        
        logger.info("Successfully created DataLoader from CSV data")
        
        # Example with DataFrame
        df = pd.DataFrame({
            "timestamp": pd.date_range(start="2024-01-01", periods=1000, freq="H"),
            "temperature": np.random.normal(25, 5, 1000),
            "humidity": np.random.normal(60, 10, 1000),
            "pressure": np.random.normal(1013, 5, 1000)
        })
        
        sequences = preprocessor.preprocess(df)
        logger.info("Successfully preprocessed DataFrame")
        
    except Exception as e:
        logger.error(f"Error in example usage: {str(e)}")
        raise
