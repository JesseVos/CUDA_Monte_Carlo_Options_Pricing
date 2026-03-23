#pragma once

/// Barrier option types.
///
/// The barrier level B divides the price space into "safe" and "hit" regions.
///
///   UpAndOut:   knocked out (pays 0) when S rises above B.
///   DownAndOut: knocked out (pays 0) when S falls below B.
///   UpAndIn:    activates (pays European payoff) only if S rises above B.
///   DownAndIn:  activates (pays European payoff) only if S falls below B.
///
/// Parity: UpAndIn + UpAndOut = European;  DownAndIn + DownAndOut = European.
enum class BarrierType {
    UpAndOut,
    DownAndOut,
    UpAndIn,
    DownAndIn
};
