// TennisIntelligenceApp.swift
// TennisIQ — AI-First Tennis Match Intelligence Platform
// Main app entry point

import SwiftUI

@main
struct TennisIntelligenceApp: App {
    @StateObject private var sessionManager = SessionManager()
    @StateObject private var subscriptionManager = SubscriptionManager()
    
    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(sessionManager)
                .environmentObject(subscriptionManager)
        }
    }
}

// MARK: - Content View (Root Navigation)
struct ContentView: View {
    @EnvironmentObject var sessionManager: SessionManager
    @State private var selectedTab = 0
    
    var body: some View {
        TabView(selection: $selectedTab) {
            MatchSetupView()
                .tabItem {
                    Label("Play", systemImage: "figure.tennis")
                }
                .tag(0)
            
            SessionHistoryView()
                .tabItem {
                    Label("History", systemImage: "clock.fill")
                }
                .tag(1)
            
            StatsView()
                .tabItem {
                    Label("Stats", systemImage: "chart.bar.fill")
                }
                .tag(2)
            
            CoachingView()
                .tabItem {
                    Label("Coach", systemImage: "brain.head.profile")
                }
                .tag(3)
            
            ProfileView()
                .tabItem {
                    Label("Profile", systemImage: "person.fill")
                }
                .tag(4)
        }
        .accentColor(.green)
    }
}

// MARK: - Placeholder Views
struct SessionHistoryView: View {
    var body: some View {
        NavigationView {
            List { Text("Session history will appear here") }
                .navigationTitle("Match History")
        }
    }
}

struct CoachingView: View {
    var body: some View {
        NavigationView {
            VStack {
                Text("AI Coaching Insights")
                    .font(.largeTitle)
                Text("Your personalized coaching feedback will appear here after each session.")
                    .multilineTextAlignment(.center)
                    .padding()
            }
            .navigationTitle("AI Coach")
        }
    }
}

struct ProfileView: View {
    var body: some View {
        NavigationView {
            List {
                Section("Player Info") {
                    Text("Name: Player")
                    Text("Handedness: Right")
                    Text("Level: Intermediate")
                }
                Section("Subscription") {
                    Text("Current Plan: Free")
                    Button("Upgrade to Pro") { }
                }
            }
            .navigationTitle("Profile")
        }
    }
}

// MARK: - Session Manager
class SessionManager: ObservableObject {
    @Published var currentSession: CaptureSessionModel?
    @Published var isRecording = false
    @Published var isCalibrated = false
    
    func startSession(mode: SessionMode, config: MatchConfiguration) {
        currentSession = CaptureSessionModel(mode: mode, config: config)
        isRecording = true
    }
    
    func stopSession() {
        isRecording = false
    }
}

// MARK: - Subscription Manager
class SubscriptionManager: ObservableObject {
    @Published var currentTier: SubscriptionTier = .free
    
    func canUseFeature(_ feature: String) -> Bool {
        switch currentTier {
        case .free: return ["line_challenge", "basic_scoring"].contains(feature)
        case .pro: return true
        case .elite: return true
        }
    }
}

// MARK: - Enums
enum SessionMode: String, CaseIterable {
    case match = "Match"
    case practice = "Practice"
    case drill = "Drill"
}

enum SubscriptionTier: String {
    case free = "Free"
    case pro = "Pro"
    case elite = "Elite"
}

// MARK: - Models
struct MatchConfiguration {
    var format: String = "Best of 3"
    var surface: String = "Hard"
    var isSingles: Bool = true
    var noAdScoring: Bool = false
    var player1Name: String = "Player 1"
    var player2Name: String = "Player 2"
}

struct CaptureSessionModel: Identifiable {
    let id = UUID()
    let mode: SessionMode
    let config: MatchConfiguration
    var isActive = true
}
