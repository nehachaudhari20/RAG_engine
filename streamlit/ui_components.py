# streamlit/ui_components.py
import streamlit as st
from rag_engine.schema.block import Block


def render_chunk(block: Block):
    st.markdown(f"**Section:** {block.section_title}")
    st.markdown(f"**Paragraphs:** {block.metadata.get('num_paragraphs', 1)}")
    st.markdown(block.preview(500))
    st.divider()
