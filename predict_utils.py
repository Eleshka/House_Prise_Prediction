# predict_utils.py
import pandas as pd
import numpy as np

def preprocess_for_prediction(df_input, preprocessor, model):
    """
    Подготавливает данные для предсказания, учитывая особенности пайплайна.
    """
    # 1. Создаем копию данных
    df = df_input.copy()
    
    # 2. Удаляем колонки, которые удаляются в my_imputer
    columns_to_drop = ['Id', 'Alley', 'PoolQC', 'Fence', 'MiscFeature']
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
    
    # 3. Выполняем импутацию вручную (так же как в обучении)
    mean_impute_numeric = ['LotFrontage', 'MasVnrArea', 'GarageYrBlt']
    mode_impute_cat = ['MasVnrType', 'Electrical']
    bsmt_cols = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']
    fireplace_cols = ['FireplaceQu']
    garage_cols = ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']
    
    # Импутация mean для числовых
    for col in mean_impute_numeric:
        if col in df.columns and df[col].isna().any():
            df[col].fillna(df[col].mean(), inplace=True)
    
    # Импутация mode для категориальных
    for col in mode_impute_cat:
        if col in df.columns and df[col].isna().any():
            df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown', inplace=True)
    
    # Импутация константами
    for col in bsmt_cols:
        if col in df.columns and df[col].isna().any():
            df[col].fillna('NB', inplace=True)
    
    for col in fireplace_cols:
        if col in df.columns and df[col].isna().any():
            df[col].fillna('NF', inplace=True)
    
    for col in garage_cols:
        if col in df.columns and df[col].isna().any():
            df[col].fillna('NG', inplace=True)
    
    # Медианная импутация для остальных числовых
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    remaining_numeric = [col for col in numeric_cols if col not in mean_impute_numeric]
    for col in remaining_numeric:
        if df[col].isna().any():
            df[col].fillna(df[col].median(), inplace=True)
    
    # Mode импутация для остальных категориальных
    cat_cols = df.select_dtypes(include=['object']).columns
    all_special_cats = mode_impute_cat + bsmt_cols + fireplace_cols + garage_cols
    remaining_cat = [col for col in cat_cols if col not in all_special_cats]
    for col in remaining_cat:
        if df[col].isna().any():
            df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown', inplace=True)
    
    return df

def get_expected_features(preprocessor):
    """
    Получает список ожидаемых признаков после препроцессинга.
    """
    # Получаем имена признаков после кодирования
    try:
        feature_names = preprocessor.named_steps['my_encoder'].get_feature_names_out()
        return list(feature_names)
    except:
        # Если не получается, возвращаем None
        return None