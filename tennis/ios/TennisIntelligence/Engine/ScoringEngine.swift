// ScoringEngine.swift
// TennisIQ — Swift port of Python scoring state machine
// On-device scoring for offline capability

import Foundation

/// Tennis point score values
enum PointScore: String, CaseIterable {
    case love = "0"
    case fifteen = "15"
    case thirty = "30"
    case forty = "40"
    case advantage = "AD"
    case game = "Game"
    
    var next: PointScore? {
        switch self {
        case .love: return .fifteen
        case .fifteen: return .thirty
        case .thirty: return .forty
        default: return nil
        }
    }
}

/// Match configuration
struct ScoringConfig {
    var bestOf: Int = 3
    var tiebreakAt: Int = 6
    var noAdScoring: Bool = false
    var finalSetTiebreak: Bool = true
    var finalSetTiebreakTo: Int = 7
}

/// Current game state
struct GameState {
    var pointsP1: PointScore = .love
    var pointsP2: PointScore = .love
    var advantagePlayer: Int? = nil  // 1 or 2
    var isTiebreak: Bool = false
    var tiebreakPointsP1: Int = 0
    var tiebreakPointsP2: Int = 0
    
    var isDeuce: Bool {
        return pointsP1 == .forty && pointsP2 == .forty && advantagePlayer == nil
    }
}

/// Set state
struct SetState {
    var gamesP1: Int = 0
    var gamesP2: Int = 0
    var setNumber: Int = 1
    var isTiebreak: Bool = false
    var isComplete: Bool = false
    var winner: Int? = nil
}

/// Full match state
struct MatchState {
    var config: ScoringConfig = ScoringConfig()
    var currentGame: GameState = GameState()
    var currentSet: SetState = SetState()
    var completedSets: [SetState] = []
    var server: Int = 1  // 1 or 2
    var isMatchOver: Bool = false
    var matchWinner: Int? = nil
    var totalPoints: Int = 0
    var pointHistory: [(winner: Int, type: String)] = []
}

/// On-device tennis scoring engine (Swift port)
class TennisScoringEngine: ObservableObject {
    @Published var state = MatchState()
    private var stateHistory: [MatchState] = []
    
    // MARK: - Match Control
    
    func startMatch(config: ScoringConfig = ScoringConfig()) {
        state = MatchState(config: config)
    }
    
    func scorePoint(winner: Int, outcomeType: String = "winner") {
        guard !state.isMatchOver else { return }
        guard winner == 1 || winner == 2 else { return }
        
        // Save state for undo
        stateHistory.append(state)
        
        state.totalPoints += 1
        state.pointHistory.append((winner: winner, type: outcomeType))
        
        if state.currentGame.isTiebreak {
            scoreTiebreakPoint(winner: winner)
        } else {
            scoreRegularPoint(winner: winner)
        }
    }
    
    func undoLastPoint() -> Bool {
        guard let previous = stateHistory.popLast() else { return false }
        state = previous
        return true
    }
    
    // MARK: - Point Scoring
    
    private func scoreRegularPoint(winner: Int) {
        let otherPlayer = winner == 1 ? 2 : 1
        let winnerScore = winner == 1 ? state.currentGame.pointsP1 : state.currentGame.pointsP2
        let otherScore = winner == 1 ? state.currentGame.pointsP2 : state.currentGame.pointsP1
        
        // Check for advantage/deuce situations
        if winnerScore == .forty && otherScore == .forty {
            if state.config.noAdScoring {
                // No-ad: sudden death at deuce
                winGame(winner: winner)
                return
            }
            if state.currentGame.advantagePlayer == winner {
                // Had advantage, wins game
                winGame(winner: winner)
                return
            } else if state.currentGame.advantagePlayer == otherPlayer {
                // Other had advantage, back to deuce
                state.currentGame.advantagePlayer = nil
                return
            } else {
                // First deuce, give advantage
                state.currentGame.advantagePlayer = winner
                return
            }
        }
        
        if winnerScore == .forty {
            winGame(winner: winner)
            return
        }
        
        // Normal point progression
        if let nextScore = winnerScore.next {
            if winner == 1 {
                state.currentGame.pointsP1 = nextScore
            } else {
                state.currentGame.pointsP2 = nextScore
            }
        }
    }
    
    private func scoreTiebreakPoint(winner: Int) {
        if winner == 1 {
            state.currentGame.tiebreakPointsP1 += 1
        } else {
            state.currentGame.tiebreakPointsP2 += 1
        }
        
        let p1 = state.currentGame.tiebreakPointsP1
        let p2 = state.currentGame.tiebreakPointsP2
        let target = state.config.finalSetTiebreakTo
        
        // Check tiebreak win (first to target with 2-point lead)
        if (p1 >= target || p2 >= target) && abs(p1 - p2) >= 2 {
            winGame(winner: p1 > p2 ? 1 : 2)
        }
        
        // Serve rotation in tiebreak (after first point, then every 2)
        let totalTBPoints = p1 + p2
        if totalTBPoints == 1 || (totalTBPoints > 1 && (totalTBPoints - 1) % 2 == 0) {
            state.server = state.server == 1 ? 2 : 1
        }
    }
    
    // MARK: - Game/Set/Match Completion
    
    private func winGame(winner: Int) {
        if winner == 1 {
            state.currentSet.gamesP1 += 1
        } else {
            state.currentSet.gamesP2 += 1
        }
        
        // Check for set win
        let g1 = state.currentSet.gamesP1
        let g2 = state.currentSet.gamesP2
        
        if (g1 >= 6 || g2 >= 6) && abs(g1 - g2) >= 2 {
            winSet(winner: g1 > g2 ? 1 : 2)
        } else if g1 == state.config.tiebreakAt && g2 == state.config.tiebreakAt {
            // Enter tiebreak
            state.currentGame = GameState(isTiebreak: true)
            state.currentSet.isTiebreak = true
            return
        }
        
        // Reset game, rotate server
        state.currentGame = GameState()
        state.server = state.server == 1 ? 2 : 1
    }
    
    private func winSet(winner: Int) {
        state.currentSet.isComplete = true
        state.currentSet.winner = winner
        state.completedSets.append(state.currentSet)
        
        // Check for match win
        let setsNeeded = (state.config.bestOf + 1) / 2
        let setsWon = state.completedSets.filter { $0.winner == winner }.count
        
        if setsWon >= setsNeeded {
            state.isMatchOver = true
            state.matchWinner = winner
            return
        }
        
        // Start new set
        state.currentSet = SetState(setNumber: state.completedSets.count + 1)
        state.currentGame = GameState()
    }
    
    // MARK: - Display
    
    func getScoreDisplay() -> String {
        let setScores = state.completedSets.map { "\($0.gamesP1)-\($0.gamesP2)" }.joined(separator: ", ")
        let currentSet = "\(state.currentSet.gamesP1)-\(state.currentSet.gamesP2)"
        let game: String
        if state.currentGame.isTiebreak {
            game = "(\(state.currentGame.tiebreakPointsP1)-\(state.currentGame.tiebreakPointsP2))"
        } else {
            game = "\(state.currentGame.pointsP1.rawValue)-\(state.currentGame.pointsP2.rawValue)"
        }
        return setScores.isEmpty ? "\(currentSet) \(game)" : "\(setScores), \(currentSet) \(game)"
    }
}
