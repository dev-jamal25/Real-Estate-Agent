"""Application constants: feature contract, bounds, and mappings."""

# The exact 12 required features for the ML model, in order
REQUIRED_FEATURES = [
    "OverallQual",
    "GrLivArea",
    "TotalBsmtSF",
    "GarageCars",
    "Neighborhood",
    "ExterQual",
    "YearBuilt",
    "FullBath",
    "KitchenQual",
    "BsmtQual",
    "TotRmsAbvGrd",
    "LotArea",
]

# Feature groups for clarity and processing
NUMERIC_FEATURES = [
    "OverallQual",
    "GrLivArea",
    "TotalBsmtSF",
    "GarageCars",
    "YearBuilt",
    "FullBath",
    "TotRmsAbvGrd",
    "LotArea",
]

# Integer numeric features (counts, years, ratings)
INTEGER_FEATURES = {
    "OverallQual",
    "GarageCars",
    "YearBuilt",
    "FullBath",
    "TotRmsAbvGrd",
}

# Float numeric features (areas, distances)
FLOAT_FEATURES = {
    "GrLivArea",
    "TotalBsmtSF",
    "LotArea",
}

ORDINAL_FEATURES = [
    "ExterQual",
    "KitchenQual",
    "BsmtQual",
]

NOMINAL_FEATURES = [
    "Neighborhood",
]

# Ordinal quality mapping: user-facing to model-facing tokens
ORDINAL_USER_TO_MODEL = {
    "None": "None",
    "Poor": "Po",
    "Fair": "Fa",
    "Typical/Average": "TA",
    "Good": "Gd",
    "Excellent": "Ex",
}

# Reverse mapping for display purposes
ORDINAL_MODEL_TO_USER = {v: k for k, v in ORDINAL_USER_TO_MODEL.items()}

# User-friendly guidance for features in follow-up messages
FEATURE_GUIDANCE = {
    "OverallQual": "overall quality from 1 to 10",
    "GrLivArea": "above-ground living area in square feet",
    "TotalBsmtSF": "basement size in square feet",
    "GarageCars": "number of cars the garage fits",
    "Neighborhood": "neighborhood name in Ames",
    "ExterQual": "exterior quality (None, Poor, Fair, Typical/Average, Good, or Excellent)",
    "YearBuilt": "year the house was built",
    "FullBath": "number of full bathrooms",
    "KitchenQual": "kitchen quality (None, Poor, Fair, Typical/Average, Good, or Excellent)",
    "BsmtQual": "basement quality (None, Poor, Fair, Typical/Average, Good, or Excellent)",
    "TotRmsAbvGrd": "total rooms above grade",
    "LotArea": "lot size in square feet",
}
