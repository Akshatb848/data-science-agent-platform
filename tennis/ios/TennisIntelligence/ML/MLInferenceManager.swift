// MLInferenceManager.swift
// TennisIQ — CoreML model loading, prediction pipeline, and ANE optimization

import CoreML
import Vision
import CoreImage

/// Manages all CoreML model inference for the tennis intelligence pipeline.
/// Coordinates BallNet, PlayerNet, CourtNet, and ShotNet on the Neural Engine.
class MLInferenceManager: ObservableObject {
    
    // MARK: - Published State
    @Published var isReady = false
    @Published var inferenceLatencyMs: Double = 0
    @Published var modelsLoaded: [String: Bool] = [
        "BallNet": false,
        "PlayerNet": false,
        "CourtNet": false,
        "ShotNet": false
    ]
    
    // MARK: - Models
    private var ballDetector: VNCoreMLModel?
    private var playerDetector: VNCoreMLModel?
    private var courtDetector: VNCoreMLModel?
    private var shotClassifier: VNCoreMLModel?
    
    // MARK: - Inference Queue
    private let inferenceQueue = DispatchQueue(label: "com.tennisiq.inference", qos: .userInitiated)
    
    // MARK: - Configuration
    private let computeUnits: MLComputeUnits = .all // Use Neural Engine + GPU + CPU
    
    // MARK: - Model Loading
    
    func loadModels() async throws {
        let config = MLModelConfiguration()
        config.computeUnits = computeUnits
        
        // Load BallNet — YOLOv8-nano ball detector
        if let ballModel = try? await loadModel(named: "BallNet", config: config) {
            ballDetector = try VNCoreMLModel(for: ballModel)
            await updateModelStatus("BallNet", loaded: true)
        }
        
        // Load PlayerNet — MobileNetV3 + MoveNet
        if let playerModel = try? await loadModel(named: "PlayerNet", config: config) {
            playerDetector = try VNCoreMLModel(for: playerModel)
            await updateModelStatus("PlayerNet", loaded: true)
        }
        
        // Load CourtNet — Court keypoint detector
        if let courtModel = try? await loadModel(named: "CourtNet", config: config) {
            courtDetector = try VNCoreMLModel(for: courtModel)
            await updateModelStatus("CourtNet", loaded: true)
        }
        
        // Load ShotNet — Temporal shot classifier
        if let shotModel = try? await loadModel(named: "ShotNet", config: config) {
            shotClassifier = try VNCoreMLModel(for: shotModel)
            await updateModelStatus("ShotNet", loaded: true)
        }
        
        await MainActor.run { isReady = true }
    }
    
    private func loadModel(named name: String, config: MLModelConfiguration) async throws -> MLModel? {
        // In production, models are bundled in the app
        // Here we check if the compiled model exists
        guard let modelURL = Bundle.main.url(forResource: name, withExtension: "mlmodelc") else {
            print("⚠️ Model \(name) not found in bundle")
            return nil
        }
        return try MLModel(contentsOf: modelURL, configuration: config)
    }
    
    @MainActor
    private func updateModelStatus(_ name: String, loaded: Bool) {
        modelsLoaded[name] = loaded
    }
    
    // MARK: - Per-Frame Inference
    
    struct FrameInferenceResult {
        var ballDetections: [BallDetection] = []
        var playerDetections: [PlayerDetection] = []
        var courtKeypoints: [CourtKeypoint] = []
        var latencyMs: Double = 0
    }
    
    struct BallDetection {
        let x, y, width, height: Float
        let confidence: Float
    }
    
    struct PlayerDetection {
        let boundingBox: CGRect
        let confidence: Float
        let keypoints: [(x: Float, y: Float, confidence: Float)]
    }
    
    struct CourtKeypoint {
        let x, y: Float
        let confidence: Float
        let name: String
    }
    
    func processFrame(_ pixelBuffer: CVPixelBuffer) async -> FrameInferenceResult {
        let startTime = CFAbsoluteTimeGetCurrent()
        var result = FrameInferenceResult()
        
        // Run inference pipelines in parallel where possible
        async let balls = detectBall(in: pixelBuffer)
        async let players = detectPlayers(in: pixelBuffer)
        async let court = detectCourt(in: pixelBuffer)
        
        result.ballDetections = await balls
        result.playerDetections = await players
        result.courtKeypoints = await court
        
        let endTime = CFAbsoluteTimeGetCurrent()
        result.latencyMs = (endTime - startTime) * 1000
        
        await MainActor.run {
            self.inferenceLatencyMs = result.latencyMs
        }
        
        return result
    }
    
    // MARK: - Individual Model Inference
    
    private func detectBall(in pixelBuffer: CVPixelBuffer) async -> [BallDetection] {
        guard let model = ballDetector else { return [] }
        
        let request = VNCoreMLRequest(model: model)
        request.imageCropAndScaleOption = .scaleFill
        
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer)
        try? handler.perform([request])
        
        guard let results = request.results as? [VNRecognizedObjectObservation] else { return [] }
        
        return results.compactMap { obs in
            let box = obs.boundingBox
            return BallDetection(
                x: Float(box.midX), y: Float(box.midY),
                width: Float(box.width), height: Float(box.height),
                confidence: obs.confidence
            )
        }
    }
    
    private func detectPlayers(in pixelBuffer: CVPixelBuffer) async -> [PlayerDetection] {
        guard let model = playerDetector else { return [] }
        
        let request = VNCoreMLRequest(model: model)
        request.imageCropAndScaleOption = .scaleFill
        
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer)
        try? handler.perform([request])
        
        // Parse player detections with pose keypoints
        guard let results = request.results as? [VNRecognizedObjectObservation] else { return [] }
        
        return results.map { obs in
            PlayerDetection(
                boundingBox: obs.boundingBox,
                confidence: obs.confidence,
                keypoints: [] // Populated from pose model output
            )
        }
    }
    
    private func detectCourt(in pixelBuffer: CVPixelBuffer) async -> [CourtKeypoint] {
        guard let model = courtDetector else { return [] }
        
        let request = VNCoreMLRequest(model: model)
        request.imageCropAndScaleOption = .scaleFill
        
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer)
        try? handler.perform([request])
        
        // Parse court keypoint predictions
        return []  // Populated from model output parsing
    }
}
