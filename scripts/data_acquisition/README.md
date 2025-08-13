# Data Acquisition Scripts

This directory contains scripts for downloading and validating the SBA loan dataset and other data sources for the AI-Powered Underwriting Engine.

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set up Kaggle API
```bash
python scripts/data_acquisition/setup_kaggle.py
```

### 3. Download and Validate SBA Dataset
```bash
python scripts/data_acquisition/run_data_acquisition.py
```

## 📋 Scripts Overview

### `setup_kaggle.py`
Sets up Kaggle API credentials for dataset downloads.

**Usage:**
```bash
python scripts/data_acquisition/setup_kaggle.py
```

**What it does:**
- Prompts for Kaggle username and API key
- Creates `~/.kaggle/kaggle.json` with proper permissions
- Verifies API connection

### `download_sba_data.py`
Downloads the SBA loan dataset from Kaggle.

**Usage:**
```bash
python scripts/data_acquisition/download_sba_data.py
```

**Features:**
- Downloads SBA dataset from Kaggle
- Extracts and organizes files
- Creates data summary with basic statistics
- Validates file integrity

### `run_data_acquisition.py`
Complete data acquisition pipeline that combines download and validation.

**Usage:**
```bash
# Full pipeline
python scripts/data_acquisition/run_data_acquisition.py

# Skip download, only validate existing data
python scripts/data_acquisition/run_data_acquisition.py --skip-download

# Use custom data directory
python scripts/data_acquisition/run_data_acquisition.py --data-dir /path/to/data
```

**Features:**
- Orchestrates complete data acquisition process
- Downloads SBA dataset
- Performs comprehensive data validation
- Generates detailed validation reports
- Calculates data quality scores

## 📊 Data Validation

The validation process includes:

- **Basic Info**: Row count, column count, memory usage
- **Missing Data**: Missing value analysis and patterns
- **Data Types**: Type validation and improvement suggestions
- **Duplicates**: Duplicate record detection
- **Outliers**: Statistical outlier identification using IQR method
- **Quality Score**: Overall data quality score (0-100)

## 📁 Output Structure

After running the acquisition pipeline:

```
data/
├── raw/
│   └── sba/
│       ├── *.csv              # Downloaded SBA datasets
│       ├── dataset_summary.json
│       └── validation_report.json
└── processed/
    └── sba/                   # For processed data (future)
```

## 🔧 Configuration

### Kaggle Credentials
- Go to https://www.kaggle.com/account
- Click "Create New API Token"
- Download `kaggle.json`
- Use `setup_kaggle.py` to configure automatically

### Dataset Sources
Currently configured for:
- **SBA Loans**: `mirbektoktogaraev/small-business-administration-sba-loans`

To add new datasets, modify the dataset name in `download_sba_data.py`.

## 🚨 Troubleshooting

### Common Issues

1. **Kaggle API Authentication Failed**
   ```
   Solution: Run setup_kaggle.py and verify credentials
   ```

2. **Dataset Not Found**
   ```
   Solution: Check dataset name and availability on Kaggle
   ```

3. **Permission Denied**
   ```
   Solution: Ensure kaggle.json has correct permissions (600)
   chmod 600 ~/.kaggle/kaggle.json
   ```

4. **Large Dataset Memory Issues**
   ```
   Solution: Use chunking for large files or increase available memory
   ```

## 📈 Next Steps

After successful data acquisition:

1. Review validation report for data quality issues
2. Run data preprocessing scripts
3. Proceed with feature engineering
4. Begin model training

## 🔗 Related Scripts

- `../init_db.py` - Database initialization
- `../../src/data_pipeline/main.py` - Main data pipeline
- `../../src/data_pipeline/preprocessing/` - Data preprocessing modules