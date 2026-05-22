# DDS-CV-MS
Building a system that detects drowsiness from a webcam by combining eye closure, blink duration, yawning, and head-nod motion, using concepts from your Computer Vision lectures instead of only a black-box CNN.

# Confidence-Aware Driver Drowsiness Detection Using KLT and Optical Flow

## Research Question

Can temporal facial motion features from KLT tracking and optical flow improve drowsiness detection compared to static eye/mouth measurements?

## Hypothesis

Drowsiness produces measurable temporal motion patterns:
- slow eyelid closing
- long eye closure
- yawning mouth expansion
- downward head nodding
- reduced facial motion stability

A confidence-aware tracker can reduce false alarms when lighting, head pose, or tracking quality changes.

## Sprint 1 Goal

Build a KLT-based facial motion tracker and log motion features from webcam video.

## Current Features

- number of tracked points
- mean horizontal displacement
- mean vertical displacement
- motion variance
- average motion speed
- forward-backward tracking error
- tracking confidence