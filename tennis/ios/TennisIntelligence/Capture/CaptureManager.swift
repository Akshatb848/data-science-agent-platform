// CaptureManager.swift
// TennisIQ — AVFoundation camera capture pipeline
// Optimized for low motion blur, deterministic timestamps, and Neural Engine inference

import AVFoundation
import CoreImage
import Combine

/// Manages the camera capture session for tennis match recording.
/// Configured for fixed FPS, low motion blur, and thermal-aware throttling.
class CaptureManager: NSObject, ObservableObject {
    
    // MARK: - Published State
    @Published var isRunning = false
    @Published var isCalibrated = false
    @Published var thermalState: ProcessInfo.ThermalState = .nominal
    @Published var currentFPS: Double = 30.0
    @Published var frameCount: Int = 0
    
    // MARK: - Capture Components
    private let captureSession = AVCaptureSession()
    private let videoOutput = AVCaptureVideoDataOutput()
    private let processingQueue = DispatchQueue(label: "com.tennisiq.capture", qos: .userInteractive)
    
    // MARK: - Configuration
    struct Config {
        var targetFPS: Double = 30.0
        var resolution: AVCaptureSession.Preset = .hd1920x1080
        var stabilization: Bool = true
        var lowLightBoost: Bool = false
        var exposureBias: Float = 0.0
        // Low motion blur: shorter exposure duration
        var maxExposureDuration: CMTime = CMTime(value: 1, timescale: 500) // 2ms
    }
    
    var config = Config()
    
    // MARK: - Callbacks
    var onFrameCaptured: ((CVPixelBuffer, CMTime, Int) -> Void)?
    
    // MARK: - Setup
    
    func setupSession() throws {
        captureSession.beginConfiguration()
        captureSession.sessionPreset = config.resolution
        
        // Camera input
        guard let camera = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back) else {
            throw CaptureError.noCameraAvailable
        }
        
        let input = try AVCaptureDeviceInput(device: camera)
        if captureSession.canAddInput(input) {
            captureSession.addInput(input)
        }
        
        // Configure camera for tennis
        try camera.lockForConfiguration()
        
        // Fixed frame rate for deterministic timestamps
        let frameDuration = CMTime(value: 1, timescale: CMTimeScale(config.targetFPS))
        camera.activeVideoMinFrameDuration = frameDuration
        camera.activeVideoMaxFrameDuration = frameDuration
        
        // Low motion blur via exposure control
        if camera.isExposureModeSupported(.custom) {
            camera.exposureMode = .custom
            camera.setExposureModeCustom(
                duration: config.maxExposureDuration,
                iso: camera.activeFormat.maxISO * 0.5
            )
        }
        
        // Optical image stabilization
        if config.stabilization && camera.activeFormat.isVideoStabilizationModeSupported(.cinematic) {
            // stabilization applied at connection level
        }
        
        // Focus
        if camera.isFocusModeSupported(.continuousAutoFocus) {
            camera.focusMode = .continuousAutoFocus
        }
        
        camera.unlockForConfiguration()
        
        // Video output
        videoOutput.videoSettings = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
        ]
        videoOutput.alwayDiscardsLateVideoFrames = true
        videoOutput.setSampleBufferDelegate(self, queue: processingQueue)
        
        if captureSession.canAddOutput(videoOutput) {
            captureSession.addOutput(videoOutput)
        }
        
        // Apply stabilization to connection
        if let connection = videoOutput.connection(with: .video) {
            if connection.isVideoStabilizationSupported {
                connection.preferredVideoStabilizationMode = .cinematic
            }
            connection.videoOrientation = .landscapeRight
        }
        
        captureSession.commitConfiguration()
        
        // Monitor thermal state
        NotificationCenter.default.addObserver(
            self, selector: #selector(thermalStateChanged),
            name: ProcessInfo.thermalStateDidChangeNotification, object: nil
        )
    }
    
    // MARK: - Session Control
    
    func startCapture() {
        processingQueue.async { [weak self] in
            self?.captureSession.startRunning()
            DispatchQueue.main.async {
                self?.isRunning = true
            }
        }
    }
    
    func stopCapture() {
        processingQueue.async { [weak self] in
            self?.captureSession.stopRunning()
            DispatchQueue.main.async {
                self?.isRunning = false
            }
        }
    }
    
    // MARK: - Thermal Management
    
    @objc private func thermalStateChanged() {
        let state = ProcessInfo.processInfo.thermalState
        DispatchQueue.main.async { [weak self] in
            self?.thermalState = state
            self?.adjustForThermalState(state)
        }
    }
    
    private func adjustForThermalState(_ state: ProcessInfo.ThermalState) {
        switch state {
        case .nominal, .fair:
            currentFPS = config.targetFPS
        case .serious:
            // Reduce to 15fps to prevent throttling
            currentFPS = min(config.targetFPS, 15)
        case .critical:
            // Minimal processing
            currentFPS = 10
        @unknown default:
            break
        }
    }
}

// MARK: - AVCaptureVideoDataOutputSampleBufferDelegate
extension CaptureManager: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(
        _ output: AVCaptureOutput,
        didOutput sampleBuffer: CMSampleBuffer,
        from connection: AVCaptureConnection
    ) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        let timestamp = CMSampleBufferGetPresentationTimeStamp(sampleBuffer)
        
        frameCount += 1
        
        // Thermal-aware frame dropping
        if thermalState == .serious && frameCount % 2 != 0 { return }
        if thermalState == .critical && frameCount % 3 != 0 { return }
        
        // Dispatch to ML inference
        onFrameCaptured?(pixelBuffer, timestamp, frameCount)
    }
}

// MARK: - Errors
enum CaptureError: Error {
    case noCameraAvailable
    case configurationFailed
    case permissionDenied
}
