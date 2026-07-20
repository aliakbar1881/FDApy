import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import freqz, tf2zpk, step, impulse, lfilter, filtfilt
import scipy.signal as sig
import json
import base64
from io import BytesIO
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Pro Filter Designer",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Professional Digital Filter Designer")
st.markdown("Design filters using **Manual** or **Auto (Wp, Ws, Ap, As)** specifications")

if 'b' not in st.session_state:
    st.session_state['b'] = None
    st.session_state['a'] = None
    st.session_state['design_info'] = ""
    st.session_state['filter_type'] = "Lowpass"
    st.session_state['fs'] = 1000.0
    st.session_state['computed_order'] = None
    if 'designs' not in st.session_state:
        st.session_state['designs'] = {}

with st.sidebar:
    st.header("🎛️ Filter Specifications")

    filter_type = st.selectbox(
        "Filter Type",
        ["Lowpass", "Highpass", "Bandpass", "Bandstop"],
        key="filter_type_select"
    )

    design_method = st.selectbox(
        "Design Method",
        ["Butterworth (IIR)", "Chebyshev Type I (IIR)", "Chebyshev Type II (IIR)",
         "Elliptic (IIR)", "Bessel (IIR)", "FIR (Window)", "FIR (Equiripple)"],
        key="design_method_select"
    )

    fs = st.number_input(
        "Sampling Frequency Fs (Hz)",
        min_value=1.0,
        value=1000.0,
        step=100.0,
        format="%.1f",
        key="fs_input"
    )

    spec_mode = st.radio(
        "Specification Mode",
        ["Manual (Order & Cutoff)", "Auto (Wp, Ws, Ap, As)"],
        key="spec_mode",
        horizontal=True
    )

    if spec_mode == "Manual (Order & Cutoff)":
        order = st.number_input(
            "Filter Order (N)",
            min_value=1,
            max_value=50,
            value=4,
            step=1,
            key="order_input"
        )

        fc = st.number_input(
            "Cutoff Frequency (Hz)",
            min_value=0.1,
            max_value=fs/2 - 0.1 if fs > 0 else 500.0,
            value=fs/4 if fs > 0 else 250.0,
            step=10.0,
            format="%.1f",
            key="fc_input"
        )

        if filter_type in ["Bandpass", "Bandstop"]:
            fc2 = st.number_input(
                "Second Cutoff Frequency (Hz)",
                min_value=fc + 1.0,
                max_value=fs/2 - 0.1 if fs > 0 else 750.0,
                value=fs * 0.4 if fs > 0 else 400.0,
                step=10.0,
                format="%.1f",
                key="fc2_input"
            )

        wn = fc / (fs / 2) if fs > 0 else 0.25
        if filter_type in ["Bandpass", "Bandstop"]:
            wn = [fc / (fs / 2), fc2 / (fs / 2)]
        wp_norm = ws_norm = None
        ap = as_db = None

    else:  # Auto Mode
        st.info("The optimal order will be computed from Wp, Ws, Ap, As")

        ap = st.number_input(
            "Ap (Passband Ripple, dB)",
            min_value=0.01,
            max_value=10.0,
            value=1.0,
            step=0.1,
            format="%.2f",
            key="ap_input"
        )

        as_db = st.number_input(
            "As (Stopband Attenuation, dB)",
            min_value=1.0,
            max_value=120.0,
            value=40.0,
            step=1.0,
            format="%.1f",
            key="as_input"
        )

        if filter_type in ["Lowpass", "Highpass"]:
            wp = st.number_input(
                "Wp (Passband Edge, Hz)",
                min_value=0.1,
                max_value=fs/2 - 0.1,
                value=fs/5 if fs > 0 else 200.0,
                step=10.0,
                format="%.1f",
                key="wp_input"
            )
            ws = st.number_input(
                "Ws (Stopband Edge, Hz)",
                min_value=wp + 1.0 if filter_type == "Lowpass" else 0.1,
                max_value=fs/2 - 0.1 if filter_type == "Lowpass" else fs/2 - 0.1,
                value=wp * 2 if filter_type == "Lowpass" and wp < fs/4 else wp / 2,
                step=10.0,
                format="%.1f",
                key="ws_input"
            )
            if filter_type == "Lowpass" and ws <= wp:
                st.warning("For Lowpass, Ws must be > Wp. Adjusting Ws.")
                ws = wp + 10.0
            if filter_type == "Highpass" and ws >= wp:
                st.warning("For Highpass, Ws must be < Wp. Adjusting Ws.")
                ws = wp - 10.0

            wp_norm = wp / (fs / 2)
            ws_norm = ws / (fs / 2)
            wn = wp_norm  # For FIR window cutoff

        else:  # Bandpass / Bandstop
            col1, col2 = st.columns(2)
            with col1:
                wp1 = st.number_input(
                    "Wp1 (Lower Passband, Hz)",
                    min_value=0.1,
                    max_value=fs/2 - 10.0,
                    value=fs/8 if fs > 0 else 100.0,
                    step=10.0,
                    format="%.1f",
                    key="wp1_input"
                )
                ws1 = st.number_input(
                    "Ws1 (Lower Stopband, Hz)",
                    min_value=0.1,
                    max_value=wp1 - 1.0,
                    value=wp1 * 0.6 if wp1 > 10 else 50.0,
                    step=10.0,
                    format="%.1f",
                    key="ws1_input"
                )
            with col2:
                wp2 = st.number_input(
                    "Wp2 (Upper Passband, Hz)",
                    min_value=wp1 + 10.0,
                    max_value=fs/2 - 0.1,
                    value=fs * 0.35 if fs > 0 else 400.0,
                    step=10.0,
                    format="%.1f",
                    key="wp2_input"
                )
                ws2 = st.number_input(
                    "Ws2 (Upper Stopband, Hz)",
                    min_value=wp2 + 1.0,
                    max_value=fs/2 - 0.1,
                    value=wp2 * 1.4 if wp2 < fs/2.5 else fs * 0.45,
                    step=10.0,
                    format="%.1f",
                    key="ws2_input"
                )

            if filter_type == "Bandpass":
                if ws1 >= wp1:
                    st.warning("Bandpass: Ws1 must be < Wp1. Adjusting.")
                    ws1 = wp1 - 10.0
                if ws2 <= wp2:
                    st.warning("Bandpass: Ws2 must be > Wp2. Adjusting.")
                    ws2 = wp2 + 10.0
            else:  # Bandstop
                if ws1 <= wp1:
                    st.warning("Bandstop: Ws1 must be > Wp1. Adjusting.")
                    ws1 = wp1 + 10.0
                if ws2 >= wp2:
                    st.warning("Bandstop: Ws2 must be < Wp2. Adjusting.")
                    ws2 = wp2 - 10.0

            wp_norm = [wp1 / (fs / 2), wp2 / (fs / 2)]
            ws_norm = [ws1 / (fs / 2), ws2 / (fs / 2)]
            wn = wp_norm  # For FIR window

    if "Chebyshev" in design_method and spec_mode == "Manual (Order & Cutoff)":
        ripple = st.slider(
            "Passband Ripple (dB)",
            min_value=0.01,
            max_value=3.0,
            value=0.5,
            step=0.01,
            key="ripple_manual"
        )
        if "Chebyshev Type II" in design_method:
            stop_atten = st.slider(
                "Stopband Attenuation (dB)",
                min_value=10,
                max_value=80,
                value=40,
                step=1,
                key="stop_atten_manual"
            )
    elif "Chebyshev" in design_method and spec_mode == "Auto (Wp, Ws, Ap, As)":
        ripple = ap
        stop_atten = as_db

    if "Elliptic" in design_method and spec_mode == "Manual (Order & Cutoff)":
        ripple = st.slider("Passband Ripple (dB)", 0.01, 3.0, 0.5, 0.01)
        stop_atten = st.slider("Stopband Attenuation (dB)", 10, 80, 40, 1)
    elif "Elliptic" in design_method and spec_mode == "Auto (Wp, Ws, Ap, As)":
        ripple = ap
        stop_atten = as_db

    if "FIR (Window)" in design_method:
        window_type = st.selectbox(
            "Window Type",
            ["hamming", "hann", "blackman", "kaiser", "bartlett", "flattop"],
            key="window_select"
        )
        if window_type == "kaiser" and spec_mode == "Manual (Order & Cutoff)":
            beta = st.slider("Kaiser Beta", 0.0, 10.0, 5.0, 0.1)
        elif window_type == "kaiser" and spec_mode == "Auto (Wp, Ws, Ap, As)":
            beta = None

    @st.cache_data
    def design_filter_pro(filter_type, design_method, spec_mode, fs,
                          order=None, wn=None, wp_norm=None, ws_norm=None,
                          ap=None, as_db=None, ripple=None, stop_atten=None,
                          window_type='hamming', beta=None):
        """Advanced filter design supporting both Manual and Auto modes."""
        b, a = None, None
        design_info = "Error"
        computed_order = None

        try:
            btype = filter_type.lower()

            if spec_mode == "Auto (Wp, Ws, Ap, As)":
                if "Butterworth" in design_method:
                    N, Wn = signal.buttord(wp_norm, ws_norm, ap, as_db)
                    b, a = signal.butter(N, Wn, btype=btype)
                    computed_order = N
                    design_info = f"Butterworth {filter_type} (Auto N={N})"

                elif "Chebyshev Type I" in design_method:
                    N, Wn = signal.cheb1ord(wp_norm, ws_norm, ap, as_db)
                    b, a = signal.cheby1(N, ap, Wn, btype=btype)
                    computed_order = N
                    design_info = f"Chebyshev I {filter_type} (Auto N={N}, Ap={ap}dB)"

                elif "Chebyshev Type II" in design_method:
                    N, Wn = signal.cheb2ord(wp_norm, ws_norm, ap, as_db)
                    b, a = signal.cheby2(N, as_db, Wn, btype=btype)
                    computed_order = N
                    design_info = f"Chebyshev II {filter_type} (Auto N={N}, As={as_db}dB)"

                elif "Elliptic" in design_method:
                    N, Wn = signal.ellipord(wp_norm, ws_norm, ap, as_db)
                    b, a = signal.ellip(N, ap, as_db, Wn, btype=btype)
                    computed_order = N
                    design_info = f"Elliptic {filter_type} (Auto N={N}, Ap={ap}dB, As={as_db}dB)"

                elif "Bessel" in design_method:
                    N, Wn = signal.buttord(wp_norm, ws_norm, ap, as_db)
                    b, a = signal.bessel(N, Wn, btype=btype, analog=False)
                    computed_order = N
                    design_info = f"Bessel {filter_type} (Auto N={N} via Buttord)"

                elif "FIR (Window)" in design_method:
                    width = abs(np.array(ws_norm) - np.array(wp_norm))
                    if isinstance(width, np.ndarray):
                        width = max(width)  # Use max width for bandpass
                    if width <= 0:
                        width = 0.1  # fallback
                    if window_type == "kaiser":
                        N, beta = signal.kaiserord(as_db, width)
                        window = ('kaiser', beta)
                    else:
                        N = int((as_db - 8) / (22 * width)) + 2
                        window = window_type
                        beta = None
                    N = max(N, 5)
                    if isinstance(wp_norm, list):
                        if filter_type == "Bandpass":
                            b = signal.firwin(N, wp_norm, pass_zero=False, window=window, fs=2.0)
                        else:
                            b = signal.firwin(N, wp_norm, pass_zero=True, window=window, fs=2.0)
                    else:
                        b = signal.firwin(N, wp_norm, window=window, fs=2.0)
                    a = [1.0]
                    computed_order = N
                    design_info = f"FIR Window {filter_type} (Auto N={N}, Window={window_type})"

                elif "FIR (Equiripple)" in design_method:
                    width = abs(np.array(ws_norm) - np.array(wp_norm))
                    if isinstance(width, np.ndarray):
                        width = max(width)
                    if width <= 0:
                        width = 0.1
                    N = int((as_db - 8) / (22 * width)) + 2
                    N = max(N, 5)
                    numtaps = N + 1 if N % 2 == 0 else N

                    if isinstance(wp_norm, list) and len(wp_norm) == 2:
                        bands = [0, ws_norm[0], wp_norm[0], wp_norm[1], ws_norm[1], 1.0]
                        if filter_type == "Bandpass":
                            desired = [0, 1, 0]
                            weight = [as_db/ap, 1, as_db/ap]
                        else:  # Bandstop
                            desired = [1, 0, 1]
                            weight = [as_db/ap, 1, as_db/ap]
                    else:
                        if filter_type == "Lowpass":
                            bands = [0, wp_norm, ws_norm, 1.0]
                            desired = [1, 0]
                            weight = [as_db/ap, 1]
                        else:  # Highpass
                            bands = [0, ws_norm, wp_norm, 1.0]
                            desired = [0, 1]
                            weight = [as_db/ap, 1]

                    b = signal.remez(numtaps, bands, desired, weight=weight, fs=2.0)
                    a = [1.0]
                    computed_order = numtaps - 1
                    design_info = f"FIR Equiripple {filter_type} (Auto N={numtaps-1})"

                return b, a, design_info, computed_order

            else:
                if "Butterworth" in design_method:
                    b, a = signal.butter(order, wn, btype=btype)
                    design_info = f"Butterworth {filter_type} (Manual N={order})"
                elif "Chebyshev Type I" in design_method:
                    b, a = signal.cheby1(order, ripple, wn, btype=btype)
                    design_info = f"Chebyshev I {filter_type} (Manual N={order}, Ripple={ripple}dB)"
                elif "Chebyshev Type II" in design_method:
                    b, a = signal.cheby2(order, stop_atten, wn, btype=btype)
                    design_info = f"Chebyshev II {filter_type} (Manual N={order}, Atten={stop_atten}dB)"
                elif "Elliptic" in design_method:
                    b, a = signal.ellip(order, ripple, stop_atten, wn, btype=btype)
                    design_info = f"Elliptic {filter_type} (Manual N={order}, Ripple={ripple}dB, Atten={stop_atten}dB)"
                elif "Bessel" in design_method:
                    b, a = signal.bessel(order, wn, btype=btype, analog=False)
                    design_info = f"Bessel {filter_type} (Manual N={order})"
                elif "FIR (Window)" in design_method:
                    if isinstance(wn, list):
                        if filter_type == "Bandpass":
                            b = signal.firwin(order + 1, wn, pass_zero=False, window=window_type)
                        else:
                            b = signal.firwin(order + 1, wn, pass_zero=True, window=window_type)
                    else:
                        b = signal.firwin(order + 1, wn, window=window_type)
                    a = [1.0]
                    design_info = f"FIR Window {filter_type} (Manual N={order}, Window={window_type})"
                elif "FIR (Equiripple)" in design_method:
                    numtaps = order + 1
                    if isinstance(wn, list):
                        bands = [0, wn[0], wn[1], 1.0]
                        if filter_type == "Bandpass":
                            desired = [0, 1, 0]
                        else:
                            desired = [1, 0, 1]
                    else:
                        bands = [0, wn, min(wn + 0.1, 1.0), 1.0]
                        if filter_type == "Lowpass":
                            desired = [1, 0]
                        else:
                            desired = [0, 1]
                    b = signal.remez(numtaps, bands, desired, fs=2.0)
                    a = [1.0]
                    design_info = f"FIR Equiripple {filter_type} (Manual N={order})"
                computed_order = order
                return b, a, design_info, computed_order

        except Exception as e:
            st.error(f"Design error: {str(e)}")
            return None, None, f"Error: {str(e)}", None

    design_clicked = st.button("🔄 Design Filter", type="primary", use_container_width=True)

    kwargs = {
        'filter_type': filter_type,
        'design_method': design_method,
        'spec_mode': spec_mode,
        'fs': fs,
        'order': order if spec_mode == "Manual (Order & Cutoff)" else None,
        'wn': wn if spec_mode == "Manual (Order & Cutoff)" else None,
        'wp_norm': wp_norm if spec_mode == "Auto (Wp, Ws, Ap, As)" else None,
        'ws_norm': ws_norm if spec_mode == "Auto (Wp, Ws, Ap, As)" else None,
        'ap': ap if spec_mode == "Auto (Wp, Ws, Ap, As)" else None,
        'as_db': as_db if spec_mode == "Auto (Wp, Ws, Ap, As)" else None,
        'ripple': ripple if 'ripple' in locals() else 0.5,
        'stop_atten': stop_atten if 'stop_atten' in locals() else 40,
        'window_type': window_type if 'window_type' in locals() else 'hamming',
        'beta': beta if 'beta' in locals() else None
    }

    if design_clicked:
        b, a, design_info, comp_order = design_filter_pro(**kwargs)
        if b is not None and a is not None:
            st.session_state['b'] = b
            st.session_state['a'] = a
            st.session_state['design_info'] = design_info
            st.session_state['filter_type'] = filter_type
            st.session_state['fs'] = fs
            st.session_state['computed_order'] = comp_order
            if comp_order is not None:
                st.success(f"✅ Designed: {design_info}")
                st.info(f"📐 Computed/Used Order: {comp_order}")
            else:
                st.success(f"✅ Designed: {design_info}")
        else:
            st.error("Filter design failed. Check parameters.")

    st.divider()
    st.header("💾 Save / Load Design")

    if st.session_state['b'] is not None:
        design_name = st.text_input("Design name", value=st.session_state['design_info'][:30], key="design_name_input")
        if st.button("💾 Save Current Design", use_container_width=True):
            design_data = {
                'b': st.session_state['b'].tolist(),
                'a': st.session_state['a'].tolist(),
                'design_info': st.session_state['design_info'],
                'filter_type': st.session_state['filter_type'],
                'fs': st.session_state['fs'],
                'computed_order': st.session_state.get('computed_order', None)
            }
            st.session_state['designs'][design_name] = design_data
            st.success(f"Design '{design_name}' saved!")
            st.rerun()

    if st.session_state['designs']:
        st.write("**Saved Designs:**")
        for name in list(st.session_state['designs'].keys()):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"📁 {name}")
            with col2:
                if st.button("Load", key=f"load_{name}"):
                    data = st.session_state['designs'][name]
                    st.session_state['b'] = np.array(data['b'])
                    st.session_state['a'] = np.array(data['a'])
                    st.session_state['design_info'] = data['design_info']
                    st.session_state['filter_type'] = data['filter_type']
                    st.session_state['fs'] = data['fs']
                    st.session_state['computed_order'] = data.get('computed_order', None)
                    st.success(f"Loaded '{name}'")
                    st.rerun()
                if st.button("🗑️", key=f"del_{name}"):
                    del st.session_state['designs'][name]
                    st.rerun()

        col_json1, col_json2 = st.columns(2)
        with col_json1:
            json_str = json.dumps(st.session_state['designs'], indent=2)
            b64 = base64.b64encode(json_str.encode()).decode()
            href = f'<a href="data:application/json;base64,{b64}" download="filter_designs.json">📥 Download All as JSON</a>'
            st.markdown(href, unsafe_allow_html=True)
        with col_json2:
            uploaded_file = st.file_uploader("Upload JSON", type="json", key="json_upload")
            if uploaded_file is not None:
                try:
                    loaded = json.load(uploaded_file)
                    st.session_state['designs'].update(loaded)
                    st.success("Designs loaded!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")

if st.session_state['b'] is not None:
    b = st.session_state['b']
    a = st.session_state['a']
    fs = st.session_state.get('fs', 1000.0)
    filter_type = st.session_state.get('filter_type', 'Lowpass')

    w, h = freqz(b, a, worN=4096)
    freq_hz = w * fs / (2 * np.pi)
    z, p, k = tf2zpk(b, a)
    t_step, y_step = step((b, a, 1))
    t_imp, y_imp = impulse((b, a, 1))

    st.subheader("🔒 Filter Stability Check")
    poles_inside = np.all(np.abs(p) < 1.0)
    if poles_inside:
        st.success("✅ **Stable** – All poles are inside the unit circle.")
    else:
        st.error("❌ **Unstable** – One or more poles are outside the unit circle. Consider reducing order or changing design method.")

    st.info(f"**Design:** {st.session_state['design_info']}  |  **Order:** {st.session_state.get('computed_order', 'N/A')}")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📈 Magnitude Response")
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        ax1.plot(freq_hz, 20 * np.log10(abs(h) + 1e-12), 'b-', linewidth=1.5)
        ax1.set_xlabel('Frequency (Hz)')
        ax1.set_ylabel('Magnitude (dB)')
        ax1.set_title('Magnitude Response')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim([0, fs/2])
        ax1.set_ylim([-80, 5])
        st.pyplot(fig1)
        plt.close(fig1)

        st.subheader("📉 Phase Response")
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.plot(freq_hz, np.unwrap(np.angle(h)), 'r-', linewidth=1.5)
        ax2.set_xlabel('Frequency (Hz)')
        ax2.set_ylabel('Phase (radians)')
        ax2.set_title('Phase Response')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim([0, fs/2])
        st.pyplot(fig2)
        plt.close(fig2)

    with col2:
        st.subheader("📍 Pole-Zero Plot")
        fig3, ax3 = plt.subplots(figsize=(6, 4))
        theta = np.linspace(0, 2*np.pi, 200)
        ax3.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.5, linewidth=0.8)
        if len(p) > 0:
            ax3.plot(np.real(p), np.imag(p), 'rx', markersize=8, label='Poles')
        if len(z) > 0:
            ax3.plot(np.real(z), np.imag(z), 'bo', markersize=6, label='Zeros')
        ax3.set_xlabel('Real Part')
        ax3.set_ylabel('Imaginary Part')
        ax3.set_title('Pole-Zero Plot')
        ax3.legend(loc='best')
        ax3.grid(True, alpha=0.3)
        ax3.axis('equal')
        ax3.set_xlim([-1.5, 1.5])
        ax3.set_ylim([-1.5, 1.5])
        st.pyplot(fig3)
        plt.close(fig3)

        st.subheader("⏱️ Step Response")
        fig4, ax4 = plt.subplots(figsize=(6, 4))
        ax4.plot(t_step, y_step, 'g-', linewidth=1.5)
        ax4.set_xlabel('Time (samples)')
        ax4.set_ylabel('Amplitude')
        ax4.set_title('Step Response')
        ax4.grid(True, alpha=0.3)
        st.pyplot(fig4)
        plt.close(fig4)

    st.subheader("💥 Impulse Response")
    fig5, ax5 = plt.subplots(figsize=(10, 3))
    ax5.stem(t_imp[:50], y_imp[:50], basefmt=" ", use_line_collection=True)
    ax5.set_xlabel('Time (samples)')
    ax5.set_ylabel('Amplitude')
    ax5.set_title('Impulse Response (first 50 samples)')
    ax5.grid(True, alpha=0.3)
    st.pyplot(fig5)
    plt.close(fig5)

    st.subheader("⏳ Group Delay")
    fig6, ax6 = plt.subplots(figsize=(10, 3))
    w_gd, gd = signal.group_delay((b, a), w=4096, fs=fs)
    ax6.plot(w_gd, gd, 'm-', linewidth=1.5)
    ax6.set_xlabel('Frequency (Hz)')
    ax6.set_ylabel('Group Delay (samples)')
    ax6.set_title('Group Delay')
    ax6.grid(True, alpha=0.3)
    ax6.set_xlim([0, fs/2])
    st.pyplot(fig6)
    plt.close(fig6)

    st.subheader("🎵 Signal Test Bench")
    with st.expander("Test Signal Settings", expanded=True):
        sig_type = st.selectbox("Signal Type", ["Sine", "Square", "Sawtooth", "Noise", "Custom"])
        duration = st.slider("Duration (seconds)", 0.1, 5.0, 1.0, 0.1)
        num_samples = int(duration * fs)

        if sig_type == "Sine":
            freq_sig = st.slider("Sine Frequency (Hz)", 10.0, fs/2, 100.0, 10.0)
            amp = st.slider("Amplitude", 0.1, 5.0, 1.0, 0.1)
            t = np.linspace(0, duration, num_samples)
            input_signal = amp * np.sin(2 * np.pi * freq_sig * t)
        elif sig_type == "Square":
            freq_sig = st.slider("Square Frequency (Hz)", 10.0, fs/2, 50.0, 10.0)
            amp = st.slider("Amplitude", 0.1, 5.0, 1.0, 0.1)
            t = np.linspace(0, duration, num_samples)
            input_signal = amp * signal.square(2 * np.pi * freq_sig * t)
        elif sig_type == "Sawtooth":
            freq_sig = st.slider("Sawtooth Frequency (Hz)", 10.0, fs/2, 50.0, 10.0)
            amp = st.slider("Amplitude", 0.1, 5.0, 1.0, 0.1)
            t = np.linspace(0, duration, num_samples)
            input_signal = amp * signal.sawtooth(2 * np.pi * freq_sig * t)
        elif sig_type == "Noise":
            amp = st.slider("Amplitude", 0.1, 5.0, 1.0, 0.1)
            input_signal = amp * np.random.randn(num_samples)
        else:
            expr = st.text_input("Expression (use 't')", "np.sin(2*np.pi*100*t) + 0.5*np.cos(2*np.pi*200*t)")
            try:
                t = np.linspace(0, duration, num_samples)
                namespace = {'t': t, 'np': np, 'sin': np.sin, 'cos': np.cos, 'pi': np.pi}
                input_signal = eval(expr, namespace)
                input_signal = np.real(input_signal)
            except Exception as e:
                st.error(f"Error: {e}")
                input_signal = np.zeros(num_samples)

        if len(input_signal) > 0:
            zero_phase = st.checkbox("Apply zero-phase filtering (filtfilt)", value=False)
            if zero_phase:
                output_signal = filtfilt(b, a, input_signal)
                label = "Zero-phase filtered"
            else:
                output_signal = lfilter(b, a, input_signal)
                label = "Filtered (lfilter)"

            fig7, ax7 = plt.subplots(figsize=(10, 4))
            ax7.plot(t, input_signal, 'b-', alpha=0.7, label='Input')
            ax7.plot(t, output_signal, 'r-', alpha=0.7, label=label)
            ax7.set_xlabel('Time (s)')
            ax7.set_ylabel('Amplitude')
            ax7.set_title('Signal Test')
            ax7.legend()
            ax7.grid(True, alpha=0.3)
            st.pyplot(fig7)
            plt.close(fig7)

    st.subheader("📤 Export Coefficients")
    col1, col2 = st.columns(2)
    with col1:
        max_len = max(len(b), len(a))
        b_pad = np.pad(b, (0, max_len - len(b)), constant_values=np.nan)
        a_pad = np.pad(a, (0, max_len - len(a)), constant_values=np.nan)
        df = pd.DataFrame({'b': b_pad, 'a': a_pad})
        csv = df.to_csv(index=False)
        b64_csv = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64_csv}" download="filter_coeffs.csv">📥 Download CSV</a>'
        st.markdown(href, unsafe_allow_html=True)
    with col2:
        st.write("**Coefficients (Top 10)**")
        df_display = pd.DataFrame({
            'Index': list(range(max_len)),
            'b': b_pad[:10],
            'a': a_pad[:10]
        })
        st.dataframe(df_display, use_container_width=True)

else:
    st.info("👈 Configure your filter and click **Design Filter** to begin.")

st.markdown("---")
st.caption("Professional Filter Designer • Supports Ap, As, Wp, Ws • Auto-Order Calculation")
