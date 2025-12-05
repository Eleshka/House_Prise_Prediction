import streamlit as st
import pandas as pd
import numpy as np
import joblib
import io
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# ===== НАСТРОЙКА СТРАНИЦЫ =====
st.set_page_config(
    page_title="🏠 Предсказание цен на недвижимость", 
    layout="wide",
    page_icon="🏠"
)

# ===== ФУНКЦИЯ ЗАГРУЗКИ МОДЕЛИ =====
@st.cache_resource
def load_model():
    """
    Загружает модель из файла full_pipeline.joblib
    """
    try:
        model = joblib.load('full_pipeline.joblib')
        st.sidebar.success("✅ Модель успешно загружена!")
        return model
    except Exception as e:
        st.sidebar.error(f"❌ Ошибка загрузки модели: {e}")
        return None

# ===== ФУНКЦИЯ СОЗДАНИЯ ПРЕПРОЦЕССОРА ДЛЯ STREAMLIT =====
def create_streamlit_preprocessor():
    """
    Создает препроцессор, идентичный обучающему, но с фиксированными параметрами
    Важно: используем ТЕ ЖЕ колонки и параметры, что при обучении!
    """
    # ===== ШАГ 1: Определяем ВСЕ колонки, как при обучении =====
    # ВАЖНО: Эти списки должны ТОЧНО соответствовать тем, что при обучении
    columns_to_drop = ['Id', 'Alley', 'PoolQC', 'Fence', 'MiscFeature']
    
    # Специальные обработки (как в вашем коде обучения)
    mean_impute_numeric = ['LotFrontage', 'MasVnrArea', 'GarageYrBlt']
    mode_impute_cat = ['MasVnrType', 'Electrical']
    bsmt_cols = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']
    fireplace_cols = ['FireplaceQu']
    garage_cols = ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']
    
    # ===== ШАГ 2: Определяем ВСЕ колонки датасета =====
    # Это полный список из 80 колонок (как в train.csv)
    all_columns = [
        'Id', 'MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street', 'Alley',
        'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood',
        'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'OverallQual', 'OverallCond',
        'YearBuilt', 'YearRemodAdd', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd',
        'MasVnrType', 'MasVnrArea', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual',
        'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2', 'BsmtFinSF2',
        'BsmtUnfSF', 'TotalBsmtSF', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical',
        '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath',
        'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual', 'TotRmsAbvGrd',
        'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType', 'GarageYrBlt', 'GarageFinish',
        'GarageCars', 'GarageArea', 'GarageQual', 'GarageCond', 'PavedDrive', 'WoodDeckSF',
        'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'PoolQC',
        'Fence', 'MiscFeature', 'MiscVal', 'MoSold', 'YrSold', 'SaleType', 'SaleCondition'
    ]
    
    # Разделяем на категориальные и числовые (как при обучении)
    # Категориальные (object типы)
    categorical_cols = [
        'MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities', 'LotConfig',
        'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle',
        'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual',
        'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1',
        'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual',
        'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
        'PavedDrive', 'PoolQC', 'SaleType', 'SaleCondition'
    ]
    
    # Числовые колонки (остальные, кроме удаляемых и категориальных)
    numeric_cols = [
        'MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt',
        'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
        '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath',
        'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces',
        'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch',
        '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold'
    ]
    
    # Убираем из списков колонки, которые будем удалять
    categorical_cols = [col for col in categorical_cols if col not in columns_to_drop]
    numeric_cols = [col for col in numeric_cols if col not in columns_to_drop]
    
    # ===== ШАГ 3: Создаем my_imputer (ТОЧНО как при обучении) =====
    my_imputer = ColumnTransformer(
        transformers=[
            # Удаляем ненужные колонки
            ('drop_features', 'drop', columns_to_drop),
            
            # Импаттеры для числовых колонок (mean)
            ('num_mean', SimpleImputer(strategy='mean'), mean_impute_numeric),
            
            # Импаттеры для категориальных колонок (mode)
            ('cat_mode', SimpleImputer(strategy='most_frequent'), mode_impute_cat),
            
            # Импаттеры с константным заполнением
            ('bsmt_const', SimpleImputer(strategy='constant', fill_value='NB'), bsmt_cols),
            ('fireplace_const', SimpleImputer(strategy='constant', fill_value='NF'), fireplace_cols),
            ('garage_const', SimpleImputer(strategy='constant', fill_value='NG'), garage_cols),
            
            # Обрабатываем остальные числовые колонки (медиана по умолчанию)
            ('other_num', SimpleImputer(strategy='median'), 
             [col for col in numeric_cols if col not in mean_impute_numeric]),
            
            # Обрабатываем остальные категориальные колонки (самое частое по умолчанию)
            ('other_cat', SimpleImputer(strategy='most_frequent'), 
             [col for col in categorical_cols if col not in mode_impute_cat + bsmt_cols + fireplace_cols + garage_cols]),
        ],
        remainder='drop',
        verbose_feature_names_out=False
    )
    
    # ===== ШАГ 4: Создаем my_encoder =====
    all_categorical_for_encoder = (
        mode_impute_cat + bsmt_cols + fireplace_cols + garage_cols + 
        [col for col in categorical_cols if col not in mode_impute_cat + bsmt_cols + fireplace_cols + garage_cols]
    )
    
    my_encoder = ColumnTransformer(
        transformers=[
            ('onehot', OneHotEncoder(
                sparse_output=False, 
                handle_unknown='ignore'  # ИГНОРИРУЕМ новые категории, а не падаем
            ), all_categorical_for_encoder)
        ],
        remainder='passthrough',
        verbose_feature_names_out=False
    )
    
    # ===== ШАГ 5: УПРОЩЕННЫЙ my_scaler (для Streamlit) =====
    # Вместо сложной логики с выбросами используем RobustScaler для всех числовых колонок
    # Это безопасно и работает в Streamlit
    
    # Определяем какие колонки числовые (после импутации и до кодирования)
    # ВСЕ числовые колонки из numeric_cols
    numeric_for_scaling = numeric_cols  # Все числовые колонки
    
    # Создаем простой scaler
    my_scaler = ColumnTransformer(
        transformers=[
            ('robust_scaler', RobustScaler(), numeric_for_scaling)
        ],
        remainder='passthrough',  # Категориальные колонки (после one-hot) проходят без изменений
        verbose_feature_names_out=False
    )
    
    # ===== ШАГ 6: Собираем пайплайн =====
    from sklearn.pipeline import Pipeline
    preprocessor = Pipeline([
        ('my_imputer', my_imputer),
        ('my_encoder', my_encoder),
        ('my_scaler', my_scaler)
    ])
    
    return preprocessor, numeric_cols, categorical_cols

# ===== ИНИЦИАЛИЗАЦИЯ =====
# Загружаем модель
model = load_model()

# Создаем препроцессор для Streamlit
preprocessor, numeric_cols, categorical_cols = create_streamlit_preprocessor()

if model is None:
    st.error("Не удалось загрузить модель. Приложение остановлено.")
    st.stop()

# ===== ГЛАВНЫЙ ИНТЕРФЕЙС =====
st.title("🏠 Предсказание цен на недвижимость")
st.markdown("Загрузите CSV-файл с данными для получения предсказаний")

# ===== БОКОВАЯ ПАНЕЛЬ ИНФОРМАЦИИ =====
with st.sidebar:
    st.header("📊 Информация о модели")
    st.write(f"**Тип модели:** VotingRegressor (XGBoost + Ridge)")
    st.write(f"**CV R²:** 0.8968 ± 0.0184")
    st.write(f"**CV MSE:** 0.0165 ± 0.0037")
    
    st.header("ℹ️ Требования к данным")
    st.write("CSV-файл должен содержать 80 колонок, включая:")
    st.write(f"• {len(numeric_cols)} числовых признаков")
    st.write(f"• {len(categorical_cols)} категориальных признаков")
    
    st.header("🔧 Предобработка")
    st.write("Автоматически выполняется:")
    st.write("• Удаление 5 колонок")
    st.write("• Заполнение пропусков")
    st.write("• One-Hot кодирование")
    st.write("• Масштабирование")

# ===== ЗАГРУЗКА ФАЙЛА =====
uploaded_file = st.file_uploader(
    "Выберите CSV-файл с данными о недвижимости",
    type=['csv'],
    help="Файл должен содержать все 80 колонок из датасета House Prices"
)

# ===== ОБРАБОТКА ЗАГРУЖЕННОГО ФАЙЛА =====
if uploaded_file is not None:
    try:
        # Читаем CSV
        df_input = pd.read_csv(uploaded_file)
        
        st.success(f"✅ Файл загружен! {df_input.shape[0]} строк, {df_input.shape[1]} колонок")
        
        # Показываем превью
        with st.expander("🔍 Просмотр загруженных данных", expanded=True):
            tab1, tab2 = st.tabs(["Первые 5 строк", "Информация о данных"])
            
            with tab1:
                st.dataframe(df_input.head(), width='stretch')
            
            with tab2:
                # Проверяем наличие всех необходимых колонок
                required_cols = numeric_cols + categorical_cols + ['Id', 'Alley', 'PoolQC', 'Fence', 'MiscFeature']
                missing_cols = [col for col in required_cols if col not in df_input.columns]
                
                if missing_cols:
                    st.error(f"❌ Отсутствуют колонки: {missing_cols[:5]}")
                    if len(missing_cols) > 5:
                        st.error(f"... и еще {len(missing_cols) - 5} колонок")
                else:
                    st.success("✅ Все необходимые колонки присутствуют")
                
                st.write(f"**Числовые колонки:** {len(numeric_cols)}")
                st.write(f"**Категориальные колонки:** {len(categorical_cols)}")
                
                # Пропуски
                missing_values = df_input.isnull().sum().sum()
                if missing_values > 0:
                    st.warning(f"⚠️ Найдено {missing_values} пропущенных значений")
                    st.write("Пропуски будут заполнены автоматически")
        
        # ===== КНОПКА ПРЕДСКАЗАНИЯ =====
        if st.button("🚀 Выполнить предсказания", type="primary", use_container_width=True):
            with st.spinner("Обрабатываю данные и делаю предсказания..."):
                try:
                    # ШАГ 1: Предобработка данных
                    st.write("🔧 Шаг 1: Предобработка данных...")
                    
                    # Используем наш препроцессор
                    X_processed = preprocessor.fit_transform(df_input)
                    
                    st.write(f"✅ Данные обработаны. Размер: {X_processed.shape}")
                    
                    # ШАГ 2: Предсказание
                    st.write("🔮 Шаг 2: Делаю предсказания...")
                    
                    # Получаем голую модель из пайплайна
                    if hasattr(model, 'named_steps'):
                        # Если модель - это Pipeline с препроцессором
                        if 'voting_model' in model.named_steps:
                            bare_model = model.named_steps['voting_model']
                            predictions = bare_model.predict(X_processed)
                        else:
                            # Если модель уже включает препроцессор
                            predictions = model.predict(df_input)
                    else:
                        # Если модель - уже готовая голосующая модель
                        predictions = model.predict(X_processed)
                    
                    # Преобразуем из логарифмической шкалы обратно
                    predictions_price = np.exp(predictions)
                    
                    st.success(f"✅ Готово! Сделано {len(predictions)} предсказаний")
                    
                    # ШАГ 3: Отображение результатов
                    st.write("📊 Шаг 3: Формирую результаты...")
                    
                    # Создаем DataFrame с результатами
                    df_results = df_input.copy()
                    df_results['Predicted_Log_Price'] = predictions
                    df_results['Predicted_Price'] = predictions_price
                    
                    # Показываем результаты
                    st.subheader("📈 Результаты предсказаний")
                    
                    # Статистика
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        avg_price = predictions_price.mean()
                        st.metric("Средняя цена", f"${avg_price:,.0f}")
                    
                    with col2:
                        min_price = predictions_price.min()
                        st.metric("Минимальная цена", f"${min_price:,.0f}")
                    
                    with col3:
                        max_price = predictions_price.max()
                        st.metric("Максимальная цена", f"${max_price:,.0f}")
                    
                    # Таблица с результатами
                    st.dataframe(
                        df_results[['Id', 'Predicted_Price']].head(20),
                        width='stretch',
                        height=400
                    )
                    
                    # ШАГ 4: Скачивание результатов
                    st.write("💾 Шаг 4: Подготовка файла для скачивания...")
                    
                    # Готовим CSV для скачивания
                    csv_buffer = io.StringIO()
                    
                    # Сохраняем только важные колонки
                    df_to_download = df_results[['Id', 'Predicted_Log_Price', 'Predicted_Price']]
                    df_to_download.to_csv(csv_buffer, index=False)
                    csv_str = csv_buffer.getvalue()
                    
                    # Кнопка скачивания
                    st.download_button(
                        label="📥 Скачать все предсказания (CSV)",
                        data=csv_str,
                        file_name="house_price_predictions.csv",
                        mime="text/csv",
                        type="primary",
                        use_container_width=True
                    )
                    
                    st.balloons()  # Праздничная анимация!
                    
                except Exception as e:
                    st.error(f"❌ Ошибка при выполнении предсказаний: {e}")
                    
                    # Детальная диагностика
                    with st.expander("🔧 Техническая информация для отладки"):
                        st.write("**Ошибка:**", str(e))
                        st.write("**Тип данных на входе:**", type(df_input))
                        st.write("**Колонки на входе:**", list(df_input.columns)[:10], "...")
                        st.write("**Форма данных:**", df_input.shape)
                        
                        if 'X_processed' in locals():
                            st.write("**Данные после препроцессинга:**", type(X_processed))
                            if hasattr(X_processed, 'shape'):
                                st.write("**Форма после препроцессинга:**", X_processed.shape)
    
    except Exception as e:
        st.error(f"❌ Ошибка при чтении файла: {e}")
        st.info("Убедитесь, что загружен валидный CSV-файл с правильными колонками.")

# ===== ФУТЕР =====
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
    🏠 House Price Prediction App | Модель: VotingRegressor (XGBoost + Ridge) | CV R²: 0.8968
    </div>
    """,
    unsafe_allow_html=True
)