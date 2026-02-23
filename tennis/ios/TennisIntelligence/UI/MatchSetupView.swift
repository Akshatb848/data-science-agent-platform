// MatchSetupView.swift
// TennisIQ — Match setup screen with court type, format, and player configuration

import SwiftUI

struct MatchSetupView: View {
    @EnvironmentObject var sessionManager: SessionManager
    @State private var mode: SessionMode = .match
    @State private var player1Name = ""
    @State private var player2Name = ""
    @State private var surface = "Hard"
    @State private var format = "Best of 3"
    @State private var isSingles = true
    @State private var noAd = false
    @State private var showingSession = false
    
    let surfaces = ["Hard", "Clay", "Grass", "Carpet"]
    let formats = ["Best of 1", "Best of 3", "Best of 5", "Tiebreak Only"]
    
    var body: some View {
        NavigationView {
            Form {
                // Session Mode
                Section("Session Type") {
                    Picker("Mode", selection: $mode) {
                        ForEach(SessionMode.allCases, id: \.self) { m in
                            Text(m.rawValue).tag(m)
                        }
                    }
                    .pickerStyle(.segmented)
                }
                
                // Players
                Section("Players") {
                    TextField("Player 1", text: $player1Name)
                    TextField("Player 2", text: $player2Name)
                    Toggle("Singles", isOn: $isSingles)
                }
                
                // Court & Format
                Section("Court & Format") {
                    Picker("Surface", selection: $surface) {
                        ForEach(surfaces, id: \.self) { Text($0) }
                    }
                    if mode == .match {
                        Picker("Format", selection: $format) {
                            ForEach(formats, id: \.self) { Text($0) }
                        }
                        Toggle("No-Ad Scoring", isOn: $noAd)
                    }
                }
                
                // Camera Setup Info
                Section("Camera Setup") {
                    HStack {
                        Image(systemName: "camera.fill")
                            .foregroundColor(.green)
                        Text("Place camera behind the baseline")
                    }
                    HStack {
                        Image(systemName: "arrow.up.and.down.and.arrow.left.and.right")
                            .foregroundColor(.orange)
                        Text("Ensure full court is visible")
                    }
                    HStack {
                        Image(systemName: "checkmark.circle.fill")
                            .foregroundColor(.blue)
                        Text("Auto-calibration will begin at start")
                    }
                }
                
                // Start Button
                Section {
                    Button(action: startSession) {
                        HStack {
                            Spacer()
                            Image(systemName: "play.fill")
                            Text("Start \(mode.rawValue)")
                                .bold()
                            Spacer()
                        }
                        .padding(.vertical, 8)
                    }
                    .foregroundColor(.white)
                    .listRowBackground(Color.green)
                }
            }
            .navigationTitle("TennisIQ")
        }
        .fullScreenCover(isPresented: $showingSession) {
            LiveOverlayView()
        }
    }
    
    func startSession() {
        let config = MatchConfiguration(
            format: format,
            surface: surface,
            isSingles: isSingles,
            noAdScoring: noAd,
            player1Name: player1Name.isEmpty ? "Player 1" : player1Name,
            player2Name: player2Name.isEmpty ? "Player 2" : player2Name
        )
        sessionManager.startSession(mode: mode, config: config)
        showingSession = true
    }
}

// MARK: - Live Session Overlay
struct LiveOverlayView: View {
    @EnvironmentObject var sessionManager: SessionManager
    @Environment(\.dismiss) var dismiss
    @State private var score = "0-0"
    @State private var shotSpeed: Double = 0
    @State private var isCalibrating = true
    
    var body: some View {
        ZStack {
            // Camera preview would go here
            Color.black.ignoresSafeArea()
            
            VStack {
                // Top bar — Score
                HStack {
                    VStack(alignment: .leading) {
                        Text(sessionManager.currentSession?.config.player1Name ?? "P1")
                            .font(.headline).foregroundColor(.white)
                        Text("0").font(.system(size: 28, weight: .bold))
                            .foregroundColor(.green)
                    }
                    Spacer()
                    VStack {
                        Text("SET 1").font(.caption).foregroundColor(.gray)
                        Text(score).font(.title2.bold()).foregroundColor(.white)
                    }
                    Spacer()
                    VStack(alignment: .trailing) {
                        Text(sessionManager.currentSession?.config.player2Name ?? "P2")
                            .font(.headline).foregroundColor(.white)
                        Text("0").font(.system(size: 28, weight: .bold))
                            .foregroundColor(.white)
                    }
                }
                .padding()
                .background(.ultraThinMaterial)
                
                Spacer()
                
                // Calibration overlay
                if isCalibrating {
                    VStack(spacing: 16) {
                        ProgressView()
                            .scaleEffect(1.5)
                            .tint(.green)
                        Text("Calibrating Court...")
                            .font(.headline)
                            .foregroundColor(.white)
                        Text("Keep camera steady")
                            .font(.subheadline)
                            .foregroundColor(.gray)
                    }
                    .padding(32)
                    .background(.ultraThinMaterial)
                    .cornerRadius(16)
                }
                
                Spacer()
                
                // Bottom controls
                HStack(spacing: 40) {
                    Button(action: { dismiss() }) {
                        VStack {
                            Image(systemName: "stop.fill")
                                .font(.title2)
                            Text("Stop").font(.caption)
                        }
                        .foregroundColor(.red)
                    }
                    
                    Button(action: { /* Challenge */ }) {
                        VStack {
                            Image(systemName: "eye.fill")
                                .font(.title2)
                            Text("Challenge").font(.caption)
                        }
                        .foregroundColor(.yellow)
                    }
                    
                    Button(action: { /* Stats */ }) {
                        VStack {
                            Image(systemName: "chart.bar.fill")
                                .font(.title2)
                            Text("Stats").font(.caption)
                        }
                        .foregroundColor(.blue)
                    }
                }
                .padding()
                .background(.ultraThinMaterial)
            }
        }
        .onAppear {
            DispatchQueue.main.asyncAfter(deadline: .now() + 3) {
                isCalibrating = false
            }
        }
    }
}

// MARK: - Stats View
struct StatsView: View {
    var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: 20) {
                    // Serve Stats Card
                    StatCard(title: "Serve", items: [
                        ("1st Serve %", "62%"),
                        ("Aces", "3"),
                        ("Double Faults", "1"),
                        ("Avg Speed", "98 mph"),
                    ])
                    
                    // Rally Stats Card
                    StatCard(title: "Rally", items: [
                        ("Winners", "12"),
                        ("Unforced Errors", "8"),
                        ("W/UE Ratio", "1.5"),
                        ("Avg Rally Length", "4.2"),
                    ])
                    
                    // Break Points Card
                    StatCard(title: "Pressure", items: [
                        ("Break Points Won", "3/5"),
                        ("Break Points Saved", "2/3"),
                        ("Tiebreaks Won", "1/1"),
                    ])
                }
                .padding()
            }
            .navigationTitle("Match Stats")
        }
    }
}

struct StatCard: View {
    let title: String
    let items: [(String, String)]
    
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text(title)
                .font(.headline)
                .foregroundColor(.green)
            
            ForEach(items, id: \.0) { label, value in
                HStack {
                    Text(label)
                        .foregroundColor(.secondary)
                    Spacer()
                    Text(value)
                        .bold()
                }
            }
        }
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(12)
    }
}
