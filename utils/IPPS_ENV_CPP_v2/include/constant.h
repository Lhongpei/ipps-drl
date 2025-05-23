// constant.h
#pragma once
// constants
inline constexpr int INF = int(1e9);
inline constexpr double EPS = 1e-6;
inline constexpr bool OR_PEER = true;
inline constexpr int HAS_NOT_SET = -1;

//Ope Type
inline constexpr int COMMON = 0;
inline constexpr int START = 1;
inline constexpr int END = 2;
inline constexpr int SUPERNODE = 3;

//Status
inline constexpr int UNSCHEDULED = 0;
inline constexpr int PROCESSING = 1;
inline constexpr int FINISHED = 2;
inline constexpr int IDLE = 3;
inline constexpr int UNFEASIBLE = 4;//Never be scheduled
inline constexpr int FEASIBLE = 5;

//Max number 
inline constexpr int MAX_OPES = 1000;  
inline constexpr int MAX_JOBS = 100; 
inline constexpr int MAX_COMBS = 2000; 
inline constexpr int MAX_MASS = 100;

//Greedy Rules
inline constexpr int EFT = 0;
inline constexpr int SPT = 1;
inline constexpr int LUM = 2;
inline constexpr int MOR = 3;
inline constexpr int FIFO = 4;
inline constexpr int MWKR = 5;
inline constexpr int RANDOM = 6;