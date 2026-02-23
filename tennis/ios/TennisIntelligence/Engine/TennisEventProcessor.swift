// TennisEventProcessor.swift
// TennisIQ — On-device CV output → tennis event semantics
// Swift port of Python event processor for offline/low-latency operation

import Foundation
import CoreGraphics

/// ITF court dimensions in meters
struct CourtDimensions {
    static let lengthHalf: CGFloat = 11.885
    static let widthSingles: CGFloat = 4.115
    static let widthDoubles: CGFloat = 5.485
    static let serviceLineDistance: CGFloat = 6.4
    static let serviceBoxWidth: CGFloat = 4.115  // center to sideline
    static let netHeight: CGFloat = 0.914
}

/// Court zone for shot placement analysis
enum CourtZone: String {
    case netLeft = "net_left"
    case netCenter = "net_center"
    case netRight = "net_right"
    case midLeft = "mid_left"
    case midCenter = "mid_center"
    case midRight = "mid_right"
    case baselineLeft = "baseline_left"
    case baselineCenter = "baseline_center"
    case baselineRight = "baseline_right"
}

/// Line call verdict
enum LineCallVerdict: String {
    case inPlay = "in"
    case outOfBounds = "out"
    case letServe = "let"
}

/// Ball bounce event with line call
struct BounceEvent {
    let courtPosition: CGPoint
    let timestamp: TimeInterval
    let frameNumber: Int
    let verdict: LineCallVerdict
    let confidence: Double
    let distanceFromLineCm: Double
    let closestLine: String
}

/// Rally state tracking
struct RallyState {
    var isActive: Bool = false
    var shotCount: Int = 0
    var startFrame: Int = 0
    var lastHitPlayer: Int = 0
    var serveSide: String = "deuce"
    var isServe: Bool = true
}

/// On-device tennis event processor
class TennisEventProcessor: ObservableObject {
    @Published var currentRally = RallyState()
    @Published var lastBounce: BounceEvent?
    @Published var isCalibrated = false
    
    private let isDoubles: Bool
    private let sidelineX: CGFloat
    
    init(isDoubles: Bool = false) {
        self.isDoubles = isDoubles
        self.sidelineX = isDoubles ? CourtDimensions.widthDoubles : CourtDimensions.widthSingles
    }
    
    // MARK: - Line Calling
    
    func processBounce(courtX: CGFloat, courtY: CGFloat, frameNumber: Int, timestamp: TimeInterval) -> BounceEvent {
        let (isIn, distCm, closestLine) = isBallIn(x: courtX, y: courtY)
        
        // Confidence based on distance from line
        let confidence: Double
        if distCm > 20 {
            confidence = 0.99
        } else if distCm > 10 {
            confidence = 0.95
        } else if distCm > 5 {
            confidence = 0.85
        } else if distCm > 2 {
            confidence = 0.70
        } else {
            confidence = 0.55
        }
        
        let bounce = BounceEvent(
            courtPosition: CGPoint(x: courtX, y: courtY),
            timestamp: timestamp,
            frameNumber: frameNumber,
            verdict: isIn ? .inPlay : .outOfBounds,
            confidence: confidence,
            distanceFromLineCm: distCm,
            closestLine: closestLine
        )
        
        lastBounce = bounce
        
        // Update rally state
        if !isIn {
            endRally()
        } else {
            currentRally.shotCount += 1
        }
        
        return bounce
    }
    
    func isBallIn(x: CGFloat, y: CGFloat, isServe: Bool = false, serveSide: String = "deuce") -> (Bool, Double, String) {
        let absX = abs(x)
        let absY = abs(y)
        
        // Distance to nearest line
        let distToSideline = (sidelineX - absX) * 100  // cm
        let distToBaseline = (CourtDimensions.lengthHalf - absY) * 100
        
        var closestLine = "sideline"
        var minDist = abs(distToSideline)
        
        if abs(distToBaseline) < minDist {
            closestLine = "baseline"
            minDist = abs(distToBaseline)
        }
        
        // Service box check for serves
        if isServe {
            let distToServiceLine = (CourtDimensions.serviceLineDistance - absY) * 100
            if abs(distToServiceLine) < minDist {
                closestLine = "service_line"
                minDist = abs(distToServiceLine)
            }
            
            let inServiceBox: Bool
            if serveSide == "deuce" {
                inServiceBox = x >= 0 && x <= sidelineX && absY <= CourtDimensions.serviceLineDistance
            } else {
                inServiceBox = x <= 0 && absX <= sidelineX && absY <= CourtDimensions.serviceLineDistance
            }
            return (inServiceBox, minDist, closestLine)
        }
        
        // Regular shot: inside baseline + sideline
        let isIn = absX <= sidelineX && absY <= CourtDimensions.lengthHalf
        return (isIn, minDist, closestLine)
    }
    
    // MARK: - Court Zone Classification
    
    func getCourtZone(x: CGFloat, y: CGFloat) -> CourtZone {
        let thirdWidth = sidelineX * 2 / 3
        let absY = abs(y)
        
        let horizontal: String
        if x < -thirdWidth / 2 {
            horizontal = "Left"
        } else if x > thirdWidth / 2 {
            horizontal = "Right"
        } else {
            horizontal = "Center"
        }
        
        let depth: String
        if absY < CourtDimensions.serviceLineDistance / 2 {
            depth = "net"
        } else if absY < CourtDimensions.serviceLineDistance {
            depth = "mid"
        } else {
            depth = "baseline"
        }
        
        let zoneString = "\(depth)_\(horizontal.lowercased())"
        return CourtZone(rawValue: zoneString) ?? .midCenter
    }
    
    // MARK: - Rally Management
    
    func startRally(frameNumber: Int) {
        currentRally = RallyState(isActive: true, startFrame: frameNumber, isServe: true)
    }
    
    func recordHit(player: Int) {
        currentRally.lastHitPlayer = player
        currentRally.shotCount += 1
        currentRally.isServe = false
    }
    
    func endRally() {
        currentRally.isActive = false
    }
    
    func toggleServeSide() {
        currentRally.serveSide = currentRally.serveSide == "deuce" ? "ad" : "deuce"
    }
}
