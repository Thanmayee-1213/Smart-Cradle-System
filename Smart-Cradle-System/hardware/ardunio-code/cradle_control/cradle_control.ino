#include <WiFi.h>
#include <AsyncTCP.h>
#include <ESPAsyncWebServer.h>
#include <ArduinoJson.h>
#include <Wire.h>
#include <Adafruit_SSD1306.h>
#include <DHT.h>
#include <ESP32Servo.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>

// ====== Pin Definitions ======
#define DHTPIN 4
#define DHTTYPE DHT11
#define MIC_PIN 36  // Analog input
#define LED_PIN 18
#define BUZZER_PIN 19
#define IR1_PIN 27
#define IR2_PIN 26
#define SERVO_SWING_PIN 14 // Servo for cry detection (mic or Flask)
#define SERVO_PAT_PIN 12   // Servo for IR detection (patting)

// ====== OLED Setup ======
#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, -1);

// ====== Sensor & Actuator Setup ======
DHT dht(DHTPIN, DHTTYPE);
Adafruit_MPU6050 mpu;
Servo servoSwing; // For cry detection
Servo servoPat;   // For IR detection (patting)

// ====== Network Setup ======
const char* ssid = "vivo1820";     // Replace with your Wi-Fi SSID
const char* password = "thanu123"; // Replace with your Wi-Fi password
AsyncWebServer server(80);

// ====== Cry Detection Parameters ======
#define MIC_SAMPLES 100
#define CRY_THRESHOLD 3095  // Adjust based on testing
#define TEMP_THRESHOLD 25.0 // Threshold temperature in Celsius
#define CRY_SWING_DURATION 3000 // 3 seconds for local cry detection

// ====== Global Variables ======
String lastCryType = "None"; // Store the latest cry type from Flask
unsigned long cradleStartTime = 0;
unsigned long cradleDuration = 0;
bool cradleActive = false;

// ====== Cry Detection Function ======
bool detectCry() {
  long sum = 0;
  for (int i = 0; i < MIC_SAMPLES; i++) {
    int val = analogRead(MIC_PIN);
    sum += val;
    delayMicroseconds(100);
  }
  int avg = sum / MIC_SAMPLES;
  return avg < CRY_THRESHOLD;
}

// ====== Servo Swing Function (Pin 14, Cry Detection) ======
void swingServo(unsigned long duration) {
  unsigned long startTime = millis();
  while (millis() - startTime < duration) {
    for (int pos = 0; pos <= 45; pos += 3) {
      servoSwing.write(pos);
      delay(15); // Normal speed
    }
    for (int pos = 45; pos >= 0; pos -= 3) {
      servoSwing.write(pos);
      delay(15);
    }
  }
  servoSwing.write(0);
}

// ====== Servo Pat Function (Pin 12, IR Detection) ======
void patServo(unsigned long duration) {
  unsigned long startTime = millis();
  while (millis() - startTime < duration) {
    for (int pos = 0; pos <= 45; pos += 3) {
      servoPat.write(pos);
      delay(30); // Slower speed for patting
    }
    for (int pos = 45; pos >= 0; pos -= 3) {
      servoPat.write(pos);
      delay(30);
    }
  }
  servoPat.write(0);
}

// ====== Update OLED Display ======
void updateDisplay(float temperature, float humidity, bool cryDetected, String cryType, bool irDetected) {
  display.clearDisplay();
  display.setTextSize(1);
  display.setTextColor(WHITE);
  display.setCursor(0, 0);

  display.print("Temp: ");
  display.print(isnan(temperature) ? String("N/A") : String(temperature));
  display.println(" C");

  display.print("Humidity: ");
  display.print(isnan(humidity) ? String("N/A") : String(humidity));
  display.println(" %");

  display.print("Cry: ");
  display.println(cryType);

  display.print("Cry Detected: ");
  display.println(cryDetected ? "Yes" : "No");

  display.print("Baby Awake: ");
  display.println(irDetected ? "Yes" : "No");

  if (!isnan(temperature) && temperature > TEMP_THRESHOLD) {
    display.println("Alert: Hot!");
  } else {
    display.println("Safe");
  }

  display.display();
}

// ====== Print Sensor Data to Serial Monitor ======
void printSensorData(float temperature, float humidity, bool cryDetected, int ir1Val, int ir2Val, int micVal) {
  Serial.print("Temperature: ");
  Serial.print(isnan(temperature) ? String("N/A") : String(temperature));
  Serial.println(" C");

  Serial.print("Humidity: ");
  Serial.print(isnan(humidity) ? String("N/A") : String(humidity));
  Serial.println(" %");

  Serial.print("Cry Detected: ");
  Serial.println(cryDetected ? "Yes" : "No");

  Serial.print("IR Sensor 1: ");
  Serial.println(ir1Val == LOW ? "Detected" : "None");

  Serial.print("IR Sensor 2: ");
  Serial.println(ir2Val == LOW ? "Detected" : "None");

  Serial.print("Baby Awake: ");
  Serial.println((ir1Val == LOW || ir2Val == LOW) ? "Yes" : "No");

  Serial.print("Microphone: ");
  Serial.println(micVal);

  Serial.print("Mic Avg: ");
  Serial.println(micVal); // Matches detectCry() output for consistency
}

void setup() {
  Serial.begin(115200);

  // Initialize pins
  pinMode(LED_PIN, OUTPUT);
  pinMode(BUZZER_PIN, OUTPUT);
  pinMode(IR1_PIN, INPUT_PULLUP);
  pinMode(IR2_PIN, INPUT_PULLUP);
  pinMode(MIC_PIN, INPUT);

  // Initialize DHT sensor
  dht.begin();

  // Initialize OLED
  if (!display.begin(SSD1306_SWITCHCAPVCC, 0x3C)) {
    Serial.println("OLED failed");
    while (1);
  }
  display.clearDisplay();
  display.display();

  // Initialize MPU6050 (optional)
  if (!mpu.begin()) {
    Serial.println("MPU6050 not found!");
  }

  // Initialize Servos
  servoSwing.attach(SERVO_SWING_PIN); // Cry detection servo
  servoPat.attach(SERVO_PAT_PIN);     // IR detection (patting) servo
  servoSwing.write(0);
  servoPat.write(0);

  // Connect to Wi-Fi
  Serial.begin(115200);
  Serial.println("Starting WiFi connection...");
  Serial.println("MAC Address: " + WiFi.macAddress());
  WiFi.begin(ssid, password);
  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 20) {
    delay(1000);
    Serial.println("Connecting to WiFi... Attempt " + String(attempts + 1) + ", Status: " + WiFi.status());
    attempts++;
  }
  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("Connected to WiFi");
    Serial.println("IP Address: " + WiFi.localIP().toString());
  } else {
    Serial.println("Failed to connect to WiFi");
    Serial.println("WiFi Status: " + String(WiFi.status()));
    Serial.println("Continuing without WiFi to test sensors...");
  }

  // ====== HTTP Server Routes ======
  // Handle prediction from Flask
  server.on("/activate_cradle", HTTP_POST, [](AsyncWebServerRequest *request) {
    if (request->hasParam("body", true)) {
      String body = request->getParam("body", true)->value();
      StaticJsonDocument<200> doc;
      DeserializationError error = deserializeJson(doc, body);
      if (error) {
        Serial.println("JSON parse error: " + String(error.c_str()));
        request->send(400, "application/json", "{\"status\":\"error\",\"error\":\"Invalid JSON\"}");
        return;
      }

      String cryType = doc["cry_type"].as<String>();
      int duration = doc["duration"].as<int>();

      // Log to Serial Monitor
      Serial.println("Flask Prediction: Cry Type = " + cryType);

      // Update cry type for OLED display
      lastCryType = cryType;

      // Only activate servo for non-not_cry predictions
      if (cryType != "not_cry") {
        cradleDuration = max((unsigned long)(duration * 1000), (unsigned long)CRY_SWING_DURATION); // Ensure at least 3 seconds
        cradleStartTime = millis();
        cradleActive = true;
      }

      // Send response
      StaticJsonDocument<100> responseDoc;
      responseDoc["status"] = "success";
      responseDoc["cry_type"] = cryType;
      String response;
      serializeJson(responseDoc, response);
      AsyncWebServerResponse *res = request->beginResponse(200, "application/json", response);
      res->addHeader("Access-Control-Allow-Origin", "*");
      request->send(res);
    } else {
      Serial.println("Error: No body provided in /activate_cradle");
      request->send(400, "application/json", "{\"status\":\"error\",\"error\":\"No body provided\"}");
    }
  });

  // Handle sensor data request
  server.on("/sensors", HTTP_GET, [](AsyncWebServerRequest *request) {
    float temperature = dht.readTemperature();
    float humidity = dht.readHumidity();
    int ir1Val = digitalRead(IR1_PIN);
    int ir2Val = digitalRead(IR2_PIN);
    int micVal = analogRead(MIC_PIN);
    bool cryDetected = detectCry();

    StaticJsonDocument<200> doc;
    doc["status"] = "success";
    doc["temperature"] = isnan(temperature) ? 0.0 : temperature;
    doc["humidity"] = isnan(humidity) ? 0.0 : humidity;
    doc["cry_detected"] = cryDetected;
    doc["ir1_value"] = ir1Val;
    doc["ir2_value"] = ir2Val;
    doc["mic_value"] = micVal;

    String response;
    serializeJson(doc, response);
    AsyncWebServerResponse *res = request->beginResponse(200, "application/json", response);
    res->addHeader("Access-Control-Allow-Origin", "*");
    request->send(res);
  });

  // Start server
  server.begin();
  Serial.println("HTTP server started");
}

void loop() {
  float temperature = dht.readTemperature();
  float humidity = dht.readHumidity();
  int ir1Val = digitalRead(IR1_PIN);
  int ir2Val = digitalRead(IR2_PIN);
  int micVal = analogRead(MIC_PIN);
  bool cryDetected = detectCry();
  bool irDetected = (ir1Val == LOW || ir2Val == LOW); // Baby awake if either IR detects

  // Print all sensor data to Serial Monitor
  printSensorData(temperature, humidity, cryDetected, ir1Val, ir2Val, micVal);

  // Update OLED with all statuses
  updateDisplay(temperature, humidity, cryDetected, lastCryType, irDetected);

  // Temperature alert
  if (!isnan(temperature) && temperature > TEMP_THRESHOLD) {
    digitalWrite(BUZZER_PIN, HIGH);
  } else {
    digitalWrite(BUZZER_PIN, LOW);
  }

  // Cradle control based on Flask prediction (swing servo, pin 14)
  if (cradleActive && (millis() - cradleStartTime < cradleDuration)) {
    Serial.println("Flask Prediction: Starting swing servo (pin 14) for " + String(cradleDuration / 1000) + " seconds");
    digitalWrite(LED_PIN, HIGH);
    swingServo(cradleDuration);
    cradleActive = false;
    digitalWrite(LED_PIN, LOW);
  }

  // Local cry detection (swing servo, pin 14)
  if (cryDetected && !cradleActive) {
    Serial.println("Local Cry Detected: Starting swing servo (pin 14) for 3 seconds");
    digitalWrite(LED_PIN, HIGH);
    swingServo(CRY_SWING_DURATION);
    digitalWrite(LED_PIN, LOW);
  }

  // IR sensor: check if baby is awake (pat servo, pin 12)
  if (irDetected && !cradleActive) {
    Serial.println("Baby is awake: Starting pat servo (pin 12) for 3 seconds");
    digitalWrite(LED_PIN, HIGH);
    patServo(CRY_SWING_DURATION);
    digitalWrite(LED_PIN, LOW);
  }

  delay(500); // Small delay before next loop
}