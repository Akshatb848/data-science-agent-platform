// WatchSessionManager.swift
// TennisIQ — Apple Watch companion for line challenges and workout tracking

import Foundation
#if os(watchOS)
import WatchConnectivity
import HealthKit

/// Manages Watch ↔ iPhone communication for tennis sessions.
class WatchSessionManager: NSObject, ObservableObject, WCSessionDelegate {
    
    @Published var isSessionActive = false
    @Published var currentScore = "0-0"
    @Published var lastLineCall: String?
    @Published var challengeResult: ChallengeResult?
    @Published var workoutActive = false
    @Published var caloriesBurned: Double = 0
    @Published var heartRate: Double = 0
    
    private var wcSession: WCSession?
    private let healthStore = HKHealthStore()
    private var workoutSession: HKWorkoutSession?
    private var workoutBuilder: HKLiveWorkoutBuilder?
    
    struct ChallengeResult {
        let verdict: String
        let confidence: Double
        let distFromLine: Double
    }
    
    // MARK: - Setup
    
    func activate() {
        if WCSession.isSupported() {
            wcSession = WCSession.default
            wcSession?.delegate = self
            wcSession?.activate()
        }
    }
    
    // MARK: - Line Challenge
    
    func requestChallenge() {
        wcSession?.sendMessage(
            ["action": "challenge", "timestamp": Date().timeIntervalSince1970],
            replyHandler: { response in
                DispatchQueue.main.async {
                    self.challengeResult = ChallengeResult(
                        verdict: response["verdict"] as? String ?? "unknown",
                        confidence: response["confidence"] as? Double ?? 0,
                        distFromLine: response["distance_cm"] as? Double ?? 0
                    )
                }
            },
            errorHandler: { error in
                print("Challenge failed: \(error)")
            }
        )
    }
    
    // MARK: - Session Control
    
    func startSession() {
        wcSession?.sendMessage(["action": "start_session"], replyHandler: nil)
        startWorkout()
    }
    
    func stopSession() {
        wcSession?.sendMessage(["action": "stop_session"], replyHandler: nil)
        endWorkout()
    }
    
    // MARK: - HealthKit Workout
    
    func startWorkout() {
        let config = HKWorkoutConfiguration()
        config.activityType = .tennis
        config.locationType = .outdoor
        
        do {
            workoutSession = try HKWorkoutSession(healthStore: healthStore, configuration: config)
            workoutBuilder = workoutSession?.associatedWorkoutBuilder()
            workoutBuilder?.dataSource = HKLiveWorkoutDataSource(healthStore: healthStore, workoutConfiguration: config)
            
            workoutSession?.startActivity(with: Date())
            workoutBuilder?.beginCollection(withStart: Date()) { success, error in
                DispatchQueue.main.async { self.workoutActive = true }
            }
        } catch {
            print("Workout start failed: \(error)")
        }
    }
    
    func endWorkout() {
        workoutSession?.end()
        workoutBuilder?.endCollection(withEnd: Date()) { success, error in
            self.workoutBuilder?.finishWorkout { workout, error in
                DispatchQueue.main.async { self.workoutActive = false }
            }
        }
    }
    
    // MARK: - WCSessionDelegate
    
    func session(_ session: WCSession, activationDidCompleteWith state: WCSessionActivationState, error: Error?) {
        DispatchQueue.main.async {
            self.isSessionActive = state == .activated
        }
    }
    
    func session(_ session: WCSession, didReceiveMessage message: [String: Any]) {
        DispatchQueue.main.async {
            if let score = message["score"] as? String {
                self.currentScore = score
            }
            if let lineCall = message["line_call"] as? String {
                self.lastLineCall = lineCall
            }
        }
    }
}
#endif
