# 📊 Exploratory Data Analysis - Bondora P2P Loans Dataset

**Dataset**: Bondora P2P Loans for AI-Powered Underwriting Engine  
**Analysis Date**: August 2025  
**Records**: 409,951 loan applications  
**Features**: 97 columns  

---

## 🎯 **Executive Summary**

Our analysis of the Bondora P2P loans dataset reveals a rich, complex dataset ideal for building an AI-powered underwriting engine. The dataset contains comprehensive loan information with strong predictive signals, though significant data quality challenges must be addressed.

### **Key Findings:**
- **Strong Target Variable**: Multiple default indicators with clear patterns
- **Rich Feature Set**: Demographics, financials, and risk scores
- **Data Quality Challenge**: 24% overall missing data requiring careful preprocessing
- **Clear Risk Patterns**: Interest rate strongly correlates with default probability (r=0.778)

---

## 📋 **Dataset Overview**

| Metric | Value |
|--------|-------|
| **Total Records** | 409,951 |
| **Total Features** | 97 |
| **Memory Usage** | 832 MB |
| **Duplicate Records** | 0 (100% unique) |
| **Missing Data** | 23.87% overall |

### **Column Distribution:**
- **Numeric Features**: 65 (67%)
- **Categorical Features**: 17 (17%)  
- **Date Features**: 15 (16%)

---

## 🎯 **Target Variable Analysis**

### **Primary Target: Loan Status**
- **Repaid**: 157,358 loans (38.4%)
- **Current**: 143,469 loans (35.0%)
- **Late**: 109,124 loans (26.6%)

### **Secondary Targets:**
- **ProbabilityOfDefault**: Mean 0.158, Range 0.0-0.994
- **DefaultDate**: Available for 124,237 loans (30.3%)
- **High-Risk Loans**: 8,741 loans (2.1%) with ProbabilityOfDefault > 0.5

### **⚠️ Key Finding:**
The dataset shows **0% explicit defaults** in completed loans, but 26.6% are "Late" status, indicating the target should be **Late payments** rather than traditional defaults.

---

## 🔍 **Missing Data Analysis**

### **Critical Issues:**
- **25 columns** have >60% missing data
- **10 columns** have 30-60% missing data  
- **22 columns** have <10% missing data

### **Most Problematic Features:**
| Feature | Missing % | Impact |
|---------|-----------|--------|
| CreditScoreEsEquifaxRisk | 97.0% | 🚨 **Unusable** |
| LoanCancelled | 95.1% | 🚨 **Unusable** |
| PlannedPrincipalTillDate | 86.7% | ⚠️ **High Impact** |
| GracePeriod* | 75.0% | ⚠️ **High Impact** |
| DefaultDate | 69.7% | ⚠️ **Expected for non-defaults** |

### **Usable Features (Low Missing):**
- All core financial features: Amount, Interest, Duration
- Demographics: Age, Country, Education
- Income variables: IncomeTotal, DebtToIncome

---

## 💰 **Financial Feature Insights**

### **Loan Characteristics:**
- **Amount**: Mean €2,489 (Std €2,037)
- **Interest Rate**: Mean 26.6% (Range 0-50%+)
- **Duration**: Primarily 60 months (63%) and 36 months (14%)
- **Monthly Payment**: Mean €95 (Std €91)

### **Risk Indicators:**
- **DebtToIncome**: Mean 2.55 (highly skewed)
- **IncomeTotal**: Mean €2,174 (highly variable)

---

## 👥 **Demographic Patterns**

### **Age Distribution:**
- **26-35**: 29.0% (largest segment)
- **36-45**: 27.5%
- **46-55**: 18.9%
- **18-25**: 10.9%
- **56+**: 13.6%

### **Geographic Distribution:**
- **Finland**: 46.2% (189,507 loans)
- **Estonia**: 43.2% (177,010 loans)  
- **Spain**: 7.6% (31,111 loans)
- **Netherlands**: 2.9% (12,027 loans)

### **Education Levels:**
- **Level 3**: 33.1% (most common)
- **Level 4**: 27.4%
- **Level 5**: 22.6%

---

## ⚖️ **Risk Assessment Features**

### **Rating Distribution:**
- **C Grade**: 30.7% (highest volume)
- **D Grade**: 20.2%
- **B Grade**: 17.7%
- **E Grade**: 11.0%
- **F Grade**: 7.0%
- **A Grades**: 9.2%
- **HR (High Risk)**: 3.6%

### **Credit Score Availability:**
- **CreditScoreEsMicroL**: 92.1% available ✅
- **CreditScoreFiAsiakasTietoRiskGrade**: 46.2% available ⚠️
- **CreditScoreEeMini**: 41.8% available ⚠️
- **CreditScoreEsEquifaxRisk**: 3.0% available 🚨

---

## 🔗 **Key Correlations**

### **Strongest Predictors of Default Risk:**
1. **Interest Rate**: r=0.778 🎯
2. **Monthly Payment**: r=0.257
3. **Debt-to-Income**: r=0.163
4. **Age**: r=-0.061 (lower risk with age)

### **Status-Based Patterns:**
| Status | Avg Amount | Avg Interest | Avg DebtToIncome | Avg Risk Score |
|--------|------------|--------------|------------------|----------------|
| Current | €2,426 | 20.6% | 0.15 | 0.10 |
| Late | €2,860 | 33.4% | 2.44 | 0.21 |
| Repaid | €2,289 | 27.4% | 4.82 | 0.17 |

---

## 🚨 **Data Quality Issues**

### **Critical Problems:**
1. **Extreme Missing Data**: 25 features >60% missing
2. **Mixed Data Types**: 6 columns have inconsistent types
3. **No Traditional Defaults**: Target requires redefinition
4. **Credit Score Gaps**: Most credit scores missing
5. **Outliers**: Age=0, extreme income values

### **Data Type Issues:**
- Columns 10, 56, 73, 74, 94, 95 have mixed types
- Requires explicit dtype specification

---

## 📈 **Preprocessing Recommendations**

### **Immediate Actions:**
1. **Drop Unusable Features**: Remove 25 features with >60% missing data
2. **Handle Mixed Types**: Fix data type inconsistencies
3. **Create Target Variable**: Use "Late" status as default indicator
4. **Feature Engineering**: Create composite risk scores
5. **Outlier Treatment**: Address extreme values (Age=0, etc.)

### **Feature Engineering Opportunities:**
1. **Risk Composite**: Combine available credit scores
2. **Income Ratios**: Payment-to-income, debt-to-income refinements
3. **Temporal Features**: Extract insights from date columns
4. **Geographic Risk**: Country-based risk factors
5. **Loan Vintage**: Time-based cohort analysis

### **Target Variable Strategy:**
```python
# Recommended target definition
target = (df['Status'] == 'Late') | df['DefaultDate'].notna()
```

---

## 🎯 **Model Development Strategy**

### **Recommended Approach:**
1. **Binary Classification**: Late/Default vs. Current/Repaid
2. **Primary Features**: Interest, Amount, Duration, Age, Income, Rating
3. **Secondary Features**: Available credit scores, geographic indicators
4. **Feature Selection**: Focus on <30% missing data features

### **Expected Performance:**
Based on correlation analysis, the model should achieve strong performance with:
- **Interest Rate** as primary predictor (r=0.778)
- **Financial ratios** as secondary predictors
- **Demographic factors** as tertiary predictors

---

## ✅ **Next Steps**

1. **Data Preprocessing**: Implement missing data strategy
2. **Feature Engineering**: Create composite variables
3. **Model Training**: Start with XGBoost on clean feature set
4. **Model Validation**: Use temporal split for realistic evaluation
5. **SHAP Analysis**: Explain model predictions for underwriting decisions

---

**📊 This EDA provides a solid foundation for building an AI-powered underwriting engine with the Bondora P2P loans dataset.**