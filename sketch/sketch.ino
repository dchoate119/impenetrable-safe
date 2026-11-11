#include "Arduino_RouterBridge.h"

const int SERVO_PIN = 9;
const int MOVE_PULSES = 40;  // 40 x 20 ms = 0.8 s of signal, then stop

volatile int target_us = 1500;
volatile int pulses_left = 0;

// Python calls servo(angle), angle 0-180
void servo(int angle) {
  angle = constrain(angle, 0, 180);
  target_us = map(angle, 0, 180, 500, 2500);
  pulses_left = MOVE_PULSES;
  Serial.print("servo -> ");
  Serial.println(angle);
}

void setup() {
  pinMode(SERVO_PIN, OUTPUT);
  digitalWrite(SERVO_PIN, LOW);
  Serial.begin(115200);
  Bridge.begin();
  Bridge.provide("servo", servo);
}

void loop() {
  if (pulses_left > 0) {
    digitalWrite(SERVO_PIN, HIGH);
    delayMicroseconds(target_us);
    digitalWrite(SERVO_PIN, LOW);
    pulses_left--;
    delay(20);
  }
}
