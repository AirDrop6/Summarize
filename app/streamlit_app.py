"""Streamlit UI."""
import streamlit as st
from app.config import settings
from app.pipeline.parser import parse_file
from app.pipeline.preprocess import clean_text
from app.pipeline.abstractive import generate_summary, load_model
from app.db.repository import (
    get_or_create_model,
    get_or_create_document,
    save_summary,
    GenerationParams,
)


st.set_page_config(page_title="Генератор научных аннотаций", layout="wide")


@st.cache_resource(show_spinner="Загрузка модели...")
def _warmup():
    load_model()
    return get_or_create_model(settings.model_name, settings.model_revision)


def main():
    st.title("Генератор аннотаций для научных статей")
    st.caption(
        f"Модель: `{settings.local_model_path or settings.model_name}` · "
        "LexRank pre-summary для длинных текстов"
    )

    model_id = _warmup()

    col_input, col_params = st.columns([3, 1])

    with col_input:
        tab_text, tab_file = st.tabs(["Текст", "Файл (pdf/docx/txt)"])

        full_text = ""
        source_type = "txt"
        filename = None

        with tab_text:
            full_text_input = st.text_area("Вставьте текст статьи", height=300)
            if full_text_input.strip():
                full_text = full_text_input
                source_type = "txt"

        with tab_file:
            uploaded = st.file_uploader(
                "Загрузите файл", type=["pdf", "docx", "txt"], accept_multiple_files=False
            )
            if uploaded is not None:
                try:
                    parsed_text, parsed_type = parse_file(uploaded.name, uploaded.read())
                    full_text = parsed_text
                    source_type = parsed_type
                    filename = uploaded.name
                    st.success(f"Извлечено {len(full_text)} символов")
                    with st.expander("Превью извлечённого текста"):
                        st.text(full_text[:2000] + ("..." if len(full_text) > 2000 else ""))
                except Exception as e:
                    st.error(f"Ошибка парсинга: {e}")

    with col_params:
        st.subheader("Параметры")
        st.caption("Длина задаётся в токенах. 1 токен ≈ 3 символа русского текста.")

        min_length = st.slider(
            "Min длина (токенов)",
            min_value=30, max_value=300, value=100, step=10,
            help="≈ символов: " + str(100 * 3) + " при значении по умолчанию",
        )
        max_length = st.slider(
            "Max длина (токенов)",
            min_value=80, max_value=500, value=250, step=10,
            help="Жёсткое ограничение сверху. Модель может закончить и раньше.",
        )
        num_beams = st.slider(
            "Beam search width", 1, 8, 6,
            help="Больше beam → лучше качество, но дольше генерация.",
        )

        st.caption(
            f"Ожидаемая длина аннотации: ~{min_length*3}–{max_length*3} символов"
        )

        if min_length >= max_length:
            st.warning("min должно быть меньше max")

    if st.button("Сгенерировать аннотацию", type="primary", disabled=not full_text.strip()):
        cleaned = clean_text(full_text)
        if len(cleaned) < 100:
            st.error("Текст слишком короткий для осмысленной аннотации.")
            return

        with st.spinner("Генерация..."):
            summary, used_extractive, latency_ms = generate_summary(
                cleaned,
                min_length=min_length,
                max_length=max_length,
                num_beams=num_beams,
            )

            # Сохраняем в БД
            doc_id = get_or_create_document(cleaned, source_type, filename)
            params = GenerationParams(
                min_length=min_length, max_length=max_length, num_beams=num_beams
            )
            save_summary(
                document_id=doc_id,
                model_id=model_id,
                summary_text=summary,
                used_extractive=used_extractive,
                latency_ms=latency_ms,
                params=params,
            )

        st.subheader("Аннотация")
        st.write(summary)

        char_count = len(summary)
        # Подсчёт токенов на выходе — для отладки/отчёта
        from app.pipeline.abstractive import load_model
        _tok, _ = load_model()
        token_count = len(_tok.encode(summary, add_special_tokens=False))

        st.caption(
            f"{latency_ms} мс · "
            f"{char_count} символов / {token_count} токенов · "
            f"запрошено {min_length}–{max_length} токенов · "
            f"{'LexRank применён' if used_extractive else ' без LexRank'}"
        )


if __name__ == "__main__":
    main()