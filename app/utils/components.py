"""MLB Toolbox — shared UI components used across multiple pages."""

import streamlit as st


def render_feedback_widget(page_name: str = "") -> None:
    """Render a shared feedback/suggestion widget at the bottom of every page."""
    with st.expander("💬 Feedback & Suggestions", expanded=False):
        _fb_type = st.radio("Type:", ["Bug Report", "Feature Request", "General Feedback"],
                            key=f"fb_type_{page_name}", horizontal=True)
        _fb_text = st.text_area("Your feedback:", key=f"fb_text_{page_name}",
                                placeholder="Describe the issue or suggestion...")
        if st.button("Submit Feedback", key=f"fb_submit_{page_name}", type="secondary"):
            if _fb_text.strip():
                st.success("Thank you! Your feedback has been recorded.")
            else:
                st.warning("Please enter some feedback text.")


def render_glossary(
    terms: list[tuple[str, str, str]],
    title: str = "\U0001f4d6 Terms & Definitions",
    cols: int = 2,
) -> None:
    """Render a collapsed expander with definition cards in a 2-column grid.

    Each entry is (abbr, full_name, description).
    """
    with st.expander(title, expanded=False):
        _gcols = st.columns(cols)
        for i, (abbr, fullname, desc) in enumerate(terms):
            with _gcols[i % cols]:
                st.markdown(
                    f"<div style='background:#0d1e35;border-left:3px solid #2b5cc8;"
                    f"border-radius:0 8px 8px 0;padding:0.5rem 0.85rem;margin-bottom:0.45rem;'>"
                    f"<span style='font-size:0.8rem;font-weight:700;color:#93c5fd;'>{abbr}</span>"
                    f"<span style='font-size:0.76rem;color:#93b8d8;'> — {fullname}</span>"
                    f"<div style='font-size:0.72rem;color:#7a9ebc;margin-top:0.15rem;line-height:1.5;'>{desc}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )


def loading_placeholder(message: str = "Loading data ...") -> None:
    """Render a centered loading card with animated progress bar."""
    st.markdown(
        f"<div class='loading-container'>"
        f"<div class='loading-icon'>⚾</div>"
        f"<div class='loading-title'>{message}</div>"
        f"<div class='loading-sub'>This may take a few seconds on first load</div>"
        f"<div class='loading-bar'></div>"
        f"</div>",
        unsafe_allow_html=True,
    )
