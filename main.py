import streamlit as st
import pandas as pd

st.title("📊 Результаты моделей машинного обучения")
st.markdown("### Сравнение производительности с разным количеством признаков")

# Результаты
columns = ["CV R2 Mean", "CV R2 Std", "CV MSE Mean", "CV MSE Std"]
features_all = [0.8967927617426292, 0.018403570040076523, 0.016533229227070893, 0.0036722856106806427]
features_16 = [0.8523954301311398, 0.02445389463691744, 0.023495813680771936, 0.004255935450286702]
features_11 = [0.8526706213121539, 0.024346487938633983, 0.023442624173625054, 0.004183919335701557]

# Create DataFrame
indexs = ['all', '16_cor', '11_cor&Permut']
df_scores = pd.DataFrame(
    [features_all, features_16, features_11],
    index=indexs,
    columns=columns
)

# Округлить до 4 знаков после запятой
df_scores = df_scores.round(4)

# Показать таблицу
st.dataframe(df_scores, use_container_width=True)

# Добавить описание
st.markdown("---")
st.markdown("**Описание метрик:**")
st.markdown("- **CV R2 Mean**: Среднее значение R² при кросс-валидации (чем выше, тем лучше)")
st.markdown("- **CV R2 Std**: Стандартное отклонение R² (чем ниже, тем стабильнее модель)")
st.markdown("- **CV MSE Mean**: Среднее значение MSE (чем ниже, тем лучше)")
st.markdown("- **CV MSE Std**: Стандартное отклонение MSE (чем ниже, тем стабильнее модель)")

# Визуализация
st.markdown("---")
st.markdown("### 📈 Визуальное сравнение")

# Выбор метрики для отображения
metric = st.selectbox("Выберите метрику для сравнения:", ["CV R2 Mean", "CV MSE Mean"])

if metric == "CV R2 Mean":
    st.bar_chart(df_scores["CV R2 Mean"])
    best_model = df_scores["CV R2 Mean"].idxmax()
    st.success(f"Лучшая модель: **{best_model}** (R² = {df_scores.loc[best_model, 'CV R2 Mean']})")
else:
    st.bar_chart(df_scores["CV MSE Mean"])
    best_model = df_scores["CV MSE Mean"].idxmin()
    st.success(f"Лучшая модель: **{best_model}** (MSE = {df_scores.loc[best_model, 'CV MSE Mean']})")

# Скачивание результатов
st.markdown("---")
st.markdown("### 💾 Скачать результаты")

# Конвертация в CSV
csv = df_scores.to_csv().encode('utf-8')
st.download_button(
    label="Скачать таблицу как CSV",
    data=csv,
    file_name="model_comparison.csv",
    mime="text/csv",
)