# FDApy

A complete, interactive **MATLAB `fdatool` (Filter Design & Analysis Tool) alternative** built entirely in Python using **Streamlit**, **SciPy**, and **Matplotlib**. Design, analyze, test, and export digital filters with a modern, browser-based GUI.

<img src="https://github.com/aliakbar1881/FDApy/blob/main/images/demo.png" alt="App Screenshot" width="800"/>

---

## ✨ Features

### 🎛️ Filter Design Specifications
- **Full Support for Industry-Standard Parameters**:
  - `Wp` (Passband edge frequency)
  - `Ws` (Stopband edge frequency)
  - `Ap` (Passband ripple in dB)
  - `As` (Stopband attenuation in dB)
- **Two Design Modes**:
  - **Manual**: Choose the order and cutoff frequency directly.
  - **Auto**: Let the app compute the **optimal order** from your `Wp`, `Ws`, `Ap`, and `As` constraints.
- **Comprehensive Filter Types**: Lowpass, Highpass, Bandpass, Bandstop.
- **Multiple Design Methods**:
  - **IIR**: Butterworth, Chebyshev Type I, Chebyshev Type II, Elliptic, Bessel.
  - **FIR**: Window-based (Hamming, Hann, Blackman, Kaiser, etc.) and Equiripple (Remez).

### 📈 Advanced Analysis & Visualization
- **Magnitude Response** (dB and linear)
- **Phase Response** (unwrapped)
- **Pole-Zero Plot** with Unit Circle
- **Impulse Response** (with visual stem plot)
- **Step Response** (rise time, overshoot visualized)
- **Group Delay**
- **Filter Stability Check**: Automatically detects if any poles fall outside the unit circle.

### 🎵 Signal Test Bench
- Generate test signals and apply your filter in real-time:
  - Sine, Square, Sawtooth, White Noise, Custom expression.
- **Zero-Phase Filtering** option (`filtfilt`) for zero-distortion filtering.
- Overlay and compare **Input vs. Output** signals visually.

### 💾 Export, Save & Load
- **Export Coefficients**: Download `b` (numerator) and `a` (denominator) as a **CSV** file.
- **Save/Load Designs**: Store filter designs locally in your browser session with named entries.
- **Import/Export as JSON**: Backup and restore your entire design library.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)

### Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/filter-designer.git
    cd filter-designer

2. **Install the required dependencies:**
```bash

pip install streamlit numpy scipy matplotlib pandas
```

3. **Running the App**

Launch the application with a single command:
```bash

streamlit run filter_designer_pro.py
```

Your default browser will automatically open to http://localhost:8501.


🧠 How to Use the App
1. Design a Filter

    Choose Filter Type: Lowpass, Highpass, Bandpass, or Bandstop.

    Select Design Method: Butterworth, Chebyshev, Elliptic, FIR Window, etc.

    Pick Specification Mode:

        Manual: Enter the filter Order and Cutoff Frequency.

        Auto: Enter Wp, Ws, Ap, and As. The app will compute the optimal order.

    Click 🔄 Design Filter. All plots will update instantly.

2. Analyze the Filter

    Check the Stability Status (green/red banner).

    Explore the Magnitude, Phase, Pole-Zero, Impulse, Step, and Group Delay plots.

    (Auto mode will display the computed filter order in the sidebar).

3. Test with Signals

    Expand the Signal Test Bench section.

    Choose a signal (e.g., Sine, Square, Noise).

    Adjust frequency, amplitude, and duration.

    Watch the Input vs. Filtered Output graph update.

    Toggle Zero-phase filtering to remove phase distortion.

4. Save & Export

    CSV: Click the download button under Export Coefficients.

    Save/Load: Enter a name in the sidebar and click Save Current Design.

    JSON: Use the Download All as JSON button to backup your entire design library.

### Possible Future Enhancements

    Quantization / Fixed-point simulation.

    HDL code generation (VHDL/Verilog).

    Real-time microphone input filtering.

    Batch design mode (design multiple filters at once).

    Frequency response specification mask overlay.
