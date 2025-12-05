import streamlit as st
import pandas as pd
import numpy as np
import joblib
import io
from predict_utils import preprocess_for_prediction, get_expected_features

# ---- Настройка страницы ----
st.set_page_config(page_title="Предсказатель цен", layout="wide")

# ---- Функция загрузки модели ----
@st.cache_resource
def load_model():
    try:
        model = joblib.load('full_pipeline.joblib')
        return model
    except Exception as e:
        st.error(f"Критическая ошибка при загрузке модели: {e}")
        return None

# ---- Загружаем модель ----
model = load_model()
if model is None:
    st.stop()

# Извлекаем препроцессор из модели (если это Pipeline)
if hasattr(model, 'named_steps') and 'preprocessor' in model.named_steps:
    preprocessor = model.named_steps['preprocessor']
else:
    # Если модель - это сам препроцессор
    preprocessor = model

# ---- Основной интерфейс ----
st.title("🏠 Предсказание цен на недвижимость")
st.markdown("Загрузите CSV-файл с данными для получения предсказаний.")

uploaded_file = st.file_uploader(
    "Выберите CSV-файл",
    type=['csv'],
    help="Файл должен содержать все необходимые для модели признаки."
)

if uploaded_file is not None:
    try:
        df_input = pd.read_csv(uploaded_file)
        st.success("✅ Файл успешно загружен!")
        
        with st.expander("🔍 Просмотр загруженных данных", expanded=True):
            st.write(f"**Размер таблицы:** {df_input.shape[0]} строк, {df_input.shape[1]} столбцов")
            st.dataframe(df_input.head(), width='stretch')
        
        # ---- ПРЕДОБРАБОТКА ДЛЯ ПРЕДСКАЗАНИЯ ----
        st.info("🔧 Выполняю предобработку данных...")
        
        # Предобрабатываем данные
        df_processed = preprocess_for_prediction(df_input, preprocessor, model)
        
        # Проверяем, что у нас есть нужные колонки
        expected_features = get_expected_features(preprocessor)
        
        if expected_features:
            st.write(f"Модель ожидает {len(expected_features)} признаков после обработки")
        
        if st.button("🚀 Выполнить предсказания", type="primary"):
            with st.spinner("Модель вычисляет предсказания..."):
                try:
                    # Делаем предсказание
                    predictions = model.predict(df_processed)
                    
                    # Обратное преобразование из логарифма (если нужно)
                    try:
                        predictions = np.exp(predictions)
                        st.info("⚠️ Предсказания были в логарифмической шкале, преобразованы обратно")
                    except:
                        pass
                    
                    # Добавляем предсказания
                    df_output = df_input.copy()
                    df_output['Predicted_SalePrice'] = predictions
                    
                    # Показываем результаты
                    st.subheader("📊 Результаты предсказаний")
                    st.dataframe(df_output[['Id', 'Predicted_SalePrice']].head(10), width='stretch')
                    
                    # Скачивание
                    csv_buffer = io.StringIO()
                    df_output.to_csv(csv_buffer, index=False)
                    csv_str = csv_buffer.getvalue()
                    
                    st.download_button(
                        label="💾 Скачать все результаты (CSV)",
                        data=csv_str,
                        file_name="house_price_predictions.csv",
                        mime="text/csv"
                    )
                    
                except Exception as e:
                    st.error(f"Ошибка при выполнении предсказаний: {e}")
                    # Детальная информация
                    st.write("**Отладочная информация:**")
                    st.write(f"Тип модели: {type(model).__name__}")
                    st.write(f"Тип данных на вход: {type(df_processed)}")
                    st.write(f"Колонки: {list(df_processed.columns)[:20]}...")
                    st.write(f"Форма: {df_processed.shape}")
                    
    except Exception as e:
        st.error(f"Ошибка при чтении файла: {e}")

# ---- Сайдбар ----
with st.sidebar:
    st.header("ℹ️ Инструкция")
    st.markdown("""
    1. Подготовьте CSV-файл с данными о доме
    2. Загрузите файл через форму
    3. Проверьте предобработанные данные
    4. Нажмите "Выполнить предсказания"
    5. Скачайте результаты
    """)
    
    # Информация о модели
    st.header("📦 Информация о модели")
    st.write(f"Тип: {type(model).__name__}")
    
    if hasattr(model, 'named_steps'):
        st.write("Шаги пайплайна:")
        for name, step in model.named_steps.items():
            st.write(f"- `{name}`: {type(step).__name__}")