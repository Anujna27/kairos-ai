"""Temporary visualization dashboard for the Kairos AI signal module.

This file is a short-term preview app and can be removed when the dashboard
team member adds their application UI.
"""

import streamlit as st
import soundfile as sf

from modules.signal_processing import process_audio


st.set_page_config(page_title="Kairos AI Audio Preview", layout="wide")
st.title("Kairos AI — Machine Sound Preview")
st.caption("Temporary dashboard for reviewing signal-processing output.")

uploaded_file = st.file_uploader("Upload a WAV audio recording", type=["wav"])

if uploaded_file is None:
    st.info("Upload a WAV file to view its waveform, spectrum, spectrogram, and extracted features.")
else:
    try:
        audio, sample_rate = sf.read(uploaded_file, dtype="float32")
        result = process_audio(audio, sample_rate)

        left, right = st.columns(2)
        with left:
            st.metric("Sample rate", f"{result['sample_rate']:,} Hz")
        with right:
            st.metric("Dominant frequency", f"{result['dominant_frequency_hz']:.1f} Hz")

        st.subheader("Waveform")
        st.plotly_chart(result["figures"]["waveform"], use_container_width=True)

        st.subheader("Spectrogram")
        st.plotly_chart(result["figures"]["spectrogram"], use_container_width=True)

        st.subheader("Frequency spectrum")
        st.plotly_chart(
            {
                "data": [{
                    "type": "scatter",
                    "mode": "lines",
                    "x": result["fft"]["frequencies"],
                    "y": result["fft"]["magnitudes"],
                    "name": "Magnitude",
                }],
                "layout": {
                    "title": "Frequency Spectrum",
                    "xaxis": {"title": "Frequency (Hz)"},
                    "yaxis": {"title": "Magnitude"},
                },
            },
            use_container_width=True,
        )

        st.subheader("Extracted acoustic and MFCC features")
        st.json(result["features"])

    except (ValueError, TypeError, RuntimeError) as error:
        st.error(f"Could not process this audio file: {error}")
