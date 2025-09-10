# 🍼 Smart Cradle System - "Lullabyte"
## 📋 Overview

An intelligent IoT-based baby monitoring system that combines artificial intelligence with embedded systems to provide autonomous infant care. The system uses machine learning for cry classification, environmental monitoring, and automated soothing responses.

## ✨ Key Features

- **🎵 AI Cry Classification** - Machine learning model identifies hunger, pain, discomfort, and tiredness
- **🌡️ Environmental Monitoring** - Real-time temperature, humidity, and air quality tracking
- **🤖 Automated Responses** - Intelligent cradle swinging and patting mechanisms
- **📱 Web Dashboard** - Real-time monitoring with mobile-responsive interface
- **🔄 Adaptive Learning** - Feedback-based model improvement system
- **👁️ Posture Monitoring** - Non-contact infant safety monitoring


## 🛠️ Hardware Components

| Component | Purpose | Pin |
|-----------|---------|-----|
| ESP32 | Main microcontroller | - |
| DHT11 | Temperature & Humidity | GPIO 4 |
| Microphone | Audio input | GPIO 36 |
| IR Sensors (2x) | Motion detection | GPIO 26, 27 |
| Servo Motors (2x) | Swing & Pat mechanism | GPIO 12, 14 |
| OLED Display | Status display | I2C |
| MPU6050 | Motion sensing | I2C |
| Buzzer | Audio alerts | GPIO 19 |
| LED | Visual indicators | GPIO 18 |

## 💻 Software Stack

- **Backend:** Python Flask, SQLite
- **Frontend:** HTML5, CSS3, JavaScript, Socket.IO
- **ML Framework:** scikit-learn, librosa
- **Hardware:** Arduino IDE, ESP32
- **Real-time Communication:** WebSocket


## 🧠 ML Model Details

### Cry Classification Categories:
- **Hungry** - Feeding required
- **Pain** - Discomfort or illness
- **Tired** - Sleep needed
- **Burping** - Gas relief needed
- **Not Cry** - Background noise

### Features Extracted:
- MFCCs (Mel-frequency cepstral coefficients)
- Spectral features (centroid, rolloff, bandwidth)
- Zero crossing rate
- RMS Energy
- Chroma features
- Tempo analysis

## 📈 Results & Performance

- **Cry Detection** - Functional classification system
- **Response Time** - <2 seconds from detection to action
- **Environmental Monitoring** - Real-time sensor data collection
- **System Reliability** - Tested for continuous operation

## 🔧 Hardware Setup

1. **Connect components** according to pin diagram
2. **Power ESP32** with appropriate 5V supply
3. **Mount sensors** securely in cradle frame
4. **Test all connections** before first use
5. **Calibrate sensors** using provided setup guide

## 📱 Web Interface Features

- **Real-time Audio Classification**
- **Environmental Data Visualization**
- **Historical Statistics & Analytics**
- **Model Performance Tracking**
- **User Feedback System**
- **Mobile-responsive Design**

## 🛡️ Safety Features

- **Environmental Monitoring** - Temperature & humidity tracking
- **Non-contact Detection** - IR-based movement sensing
- **Real-time Alerts** - Immediate notifications
- **Fail-safe Design** - Safe operation protocols

## 🔮 Future Enhancements

- [ ] Mobile app development
- [ ] Cloud integration capabilities
- [ ] Advanced ML models
- [ ] Smart home integration
- [ ] Multi-device support
- [ ] Enhanced user interface

## 🧪 Testing & Development

The system includes:
- **Prototype Testing** - Hardware validation
- **Software Testing** - Algorithm verification
- **Integration Testing** - Complete system validation
- **User Experience Testing** - Interface usability

**🍼 Making infant care technology accessible and intelligent.**


