#include <iostream>
#include <thread>
#include <vector>
#include <string>
#include <mutex>
#include <condition_variable>
#include <map>
#include <set>
#include <random>
#include <chrono>
#include <memory>        
#include <functional>    
#include <atomic>        
#include <iomanip>       
#include <numeric>       
#include <algorithm>    
#include <cstdlib>       
#include <tuple>         

// --- Constants and Configuration ---
const int NUM_TRAINS = 4;
const int SEATS_PER_CAR = 2; 
const int MIN_CARS_PER_TRAIN = 1; 
const int MAX_CARS_PER_TRAIN = 3;

const int NUM_TRACKS = 8; 
const int SIMULATION_TIME_SECONDS = 120; 
const int VISUALIZATION_REFRESH_MS = 250; 
const int TRAIN_ACTION_DELAY_MS = 500; 
const int PASSENGER_ACTION_DELAY_MS = 25; 

// Define min/max length for random routes
const int MIN_ROUTE_LENGTH = 3; // Routes should be longer to increase track overlap
const int MAX_ROUTE_LENGTH = NUM_TRACKS; // Max route length means trains will traverse most tracks

// Coal specific constants
const int MAX_COAL_CAPACITY = 100;
const int COAL_CONSUMPTION_PER_MOVE = 5;
const int COAL_REFILL_AMOUNT = 50;
const int COAL_LOW_THRESHOLD = 20; // Train will seek refill when coal is below this
const int COAL_REFILL_STATION_ID = 0; // Station 1 (0-based internally) is the coal refill station

// --- Mutexes and Condition Variables for Synchronization ---
std::mutex mtx_cout; // Mutex for safe console output

// Conflict 1: Where passenger wants to go (Route/train selection)
// Representation that passengers are waiting for a given train/route
std::mutex mtx_passengers_waiting;
std::condition_variable cv_passengers_ready;
// Map: Train ID (1-based) -> List of tuples {Passenger Name, Start Station (0-based), Destination Station (0-based)}
std::map<int, std::vector<std::tuple<std::string, int, int>>> passengers_waiting_for_train;

// Conflict 2: Occupying seats in cars
struct Car {
    std::mutex mtx_seats;
    std::vector<std::string> seats_occupied_by_passenger; // Stores passenger name or empty string if free

    Car(int num_seats) : seats_occupied_by_passenger(num_seats, "") {}

    // Explicitly delete copy constructor and copy assignment operator to ensure Car is never copied
    Car(const Car&) = delete;
    Car& operator=(const Car&) = delete;

    // Default move constructor is sufficient,
    // as std::mutex is not moved, and std::vector is.
    // We leave it as default.
    Car(Car&&) = default;
    Car& operator=(Car&&) = default;

    // Attempt to occupy a seat
    bool occupySeat(int& occupied_seat_idx, const std::string& passenger_name) {
        std::lock_guard<std::mutex> lock(mtx_seats);
        for (int i = 0; i < seats_occupied_by_passenger.size(); ++i) {
            if (seats_occupied_by_passenger[i].empty()) { // If seat is empty
                seats_occupied_by_passenger[i] = passenger_name;
                occupied_seat_idx = i;
                return true;
            }
        }
        return false; // No free seats
    }

    // Release a seat
    void releaseSeat(int seat_idx) {
        std::lock_guard<std::mutex> lock(mtx_seats);
        if (seat_idx >= 0 && seat_idx < seats_occupied_by_passenger.size()) {
            seats_occupied_by_passenger[seat_idx] = ""; // Set to empty string
        }
    }

    int getFreeSeatsCount() {
        std::lock_guard<std::mutex> lock(mtx_seats);
        int free = 0;
        for (const auto& passenger_name : seats_occupied_by_passenger) {
            if (passenger_name.empty()) {
                free++;
            }
        }
        return free;
    }

    // For visualization: Safely get the passenger name at a specific seat
    std::string getPassengerAtSeat(int seat_idx) {
        std::lock_guard<std::mutex> lock(mtx_seats);
        if (seat_idx >= 0 && seat_idx < seats_occupied_by_passenger.size()) {
            return seats_occupied_by_passenger[seat_idx];
        }
        return "";
    }
};

// Conflict 3: Signaling and track blocks
// Global vectors for mutexes and condition variables.
std::vector<std::mutex> mtx_tracks(NUM_TRACKS); // Initializes NUM_TRACKS mutexes
std::vector<std::condition_variable> cv_tracks(NUM_TRACKS); // Initializes NUM_TRACKS condition_variables

// track_occupied_by_train will store Train ID (1-based), or -1 if free.
std::vector<int> track_occupied_by_train; // GLOBAL STATE FOR VISUALIZATION (occupied status)
// track_reserved_by_train will store Train ID (1-based), or -1 if not reserved
std::vector<int> track_reserved_by_train; // GLOBAL STATE FOR VISUALIZATION (reserved status)

// Mutex and CV for the coal refill station
std::mutex mtx_coal_station;
std::condition_variable cv_coal_station_free;
bool coal_station_occupied = false;


// Flaga do bezpiecznego zatrzymywania wątków
std::atomic<bool> simulation_running(true);

// --- Reporting Structures and Globals ---
struct PassengerReportEntry {
    std::string passenger_name;
    std::chrono::steady_clock::time_point arrival_at_station_time; // When passenger started waiting
    std::chrono::steady_clock::time_point board_time;              // When passenger boarded the train
    std::chrono::steady_clock::time_point disembark_time;          // When passenger disembarked
    bool boarded_successfully = false; // True if passenger actually boarded
    bool had_to_rewait = false; // True if passenger tried to board but no seat was found
    bool reached_destination = false; // True if passenger disembarked at their intended destination
};

std::vector<PassengerReportEntry> passenger_reports;
std::mutex mtx_passenger_reports; // Mutex to protect access to passenger_reports

// For track utilization reporting
// Each entry is a pair: {start_time, end_time} for a track being occupied
std::vector<std::vector<std::pair<std::chrono::steady_clock::time_point, std::chrono::steady_clock::time_point>>> track_occupancy_intervals;
std::mutex mtx_track_occupancy_intervals; // Mutex to protect track_occupancy_intervals


// Funkcja do bezpiecznego wypisywania na konsolę
void safe_print(const std::string& message) {
    std::lock_guard<std::mutex> lock(mtx_cout);
    std::cout << message << std::endl;
}

// --- Klasy reprezentujące elementy symulacji ---

class Train {
private:
    int id; // 1-based ID
    std::vector<std::unique_ptr<Car>> cars;
    // Map: Passenger Name -> {Car Index, Seat Index, Destination Station (0-based)}
    std::map<std::string, std::tuple<int, int, int>> passengers_on_board;
    // Current track ID (0-based)
    std::atomic<int> current_track;
    std::mt19937 rng; // Generator liczb losowych
    // Route contains 0-based track IDs
    std::vector<int> route;
    size_t route_index; // Aktualny indeks na trasie
    std::atomic<int> coal_level; // Current coal level

    std::vector<long long> track_wait_times_ms; // Time spent waiting for tracks
    std::vector<long long> coal_refill_times_ms; // Time spent refilling coal
    std::vector<long long> movement_times_ms; // Time spent actively moving between tracks
    int full_route_cycles = 0; // Number of times the train completed its full route

public:
    Train(int p_id, const std::vector<int>& p_route, int num_cars)
        : id(p_id), current_track(-1), route(p_route), route_index(0), coal_level(MAX_COAL_CAPACITY) {
        for (int i = 0; i < num_cars; ++i) { // Use num_cars parameter
            cars.push_back(std::make_unique<Car>(SEATS_PER_CAR));
        }
        rng.seed(std::chrono::system_clock::now().time_since_epoch().count() + id);
    }

    Train(const Train&) = delete;
    Train& operator=(const Train&) = delete;

    Train(Train&&) = default;
    Train& operator=(Train&&) = default;

    int getId() const { return id; }
    int getCurrentTrack() const { return current_track.load(); } // Safe atomic read (0-based)
    const std::vector<int>& getRoute() const { return route; }
    int getCoalLevel() const { return coal_level.load(); }
    int getFullRouteCycles() const { return full_route_cycles; }


    // Access to cars for visualization
    const std::vector<std::unique_ptr<Car>>& getCars() const {
        return cars;
    }

    // Getters for report data
    const std::vector<long long>& getTrackWaitTimes() const { return track_wait_times_ms; }
    const std::vector<long long>& getCoalRefillTimes() const { return coal_refill_times_ms; }
    const std::vector<long long>& getMovementTimes() const { return movement_times_ms; }


    // Function to refill coal
    void refillCoal() {
        auto start_refill_time = std::chrono::steady_clock::now();
        std::unique_lock<std::mutex> lock(mtx_coal_station);
        cv_coal_station_free.wait(lock, [this] { return !coal_station_occupied || !simulation_running.load(); });

        if (!simulation_running.load()) return;

        coal_station_occupied = true;
        safe_print("[Train " + std::to_string(id) + "] is refilling coal at Station " + std::to_string(COAL_REFILL_STATION_ID + 1) + ".");
        std::this_thread::sleep_for(std::chrono::milliseconds(TRAIN_ACTION_DELAY_MS * 3)); // Simulate refill time
        coal_level = std::min(MAX_COAL_CAPACITY, coal_level.load() + COAL_REFILL_AMOUNT);
        safe_print("[Train " + std::to_string(id) + "] finished refilling coal. Current level: " + std::to_string(coal_level.load()));
        coal_station_occupied = false;
        cv_coal_station_free.notify_one();

        auto end_refill_time = std::chrono::steady_clock::now();
        coal_refill_times_ms.push_back(std::chrono::duration_cast<std::chrono::milliseconds>(end_refill_time - start_refill_time).count());
    }


    // Funkcja do symulacji ruchu pociągu i obsługi konfliktów
    void run() {
        // Output Train ID as 1-based
        safe_print("[Train " + std::to_string(id) + "] Starting! My route: " + getRouteString());
        std::this_thread::sleep_for(std::chrono::milliseconds(TRAIN_ACTION_DELAY_MS)); // Spowolnienie

        while (simulation_running.load()) {
            int next_track_0based;
            int old_current_track_0based = current_track.load(); // 0-based

            // --- DECKIEL: Logic for initial track acquisition or next track ---
            if (old_current_track_0based == -1) { // If train is in depot
                next_track_0based = route[0]; // Aim for the first station on its route
                safe_print("[Train " + std::to_string(id) + "] Attempting to enter initial Station " + std::to_string(next_track_0based + 1) + " from Depot.");
            }
            else {
                // --- Coal management ---
                if (coal_level.load() < COAL_LOW_THRESHOLD && current_track.load() != COAL_REFILL_STATION_ID) {
                    safe_print("[Train " + std::to_string(id) + "] Coal low (" + std::to_string(coal_level.load()) + "). Heading to refill station " + std::to_string(COAL_REFILL_STATION_ID + 1) + ".");
                    next_track_0based = COAL_REFILL_STATION_ID;
                }
                else if (coal_level.load() >= COAL_LOW_THRESHOLD && current_track.load() == COAL_REFILL_STATION_ID) {
                    // If at refill station and coal is topped up, return to regular route
                    route_index = (route_index + 1) % route.size();
                    next_track_0based = route[route_index];
                    safe_print("[Train " + std::to_string(id) + "] Coal topped up, returning to regular route. Next: Station " + std::to_string(next_track_0based + 1) + ".");
                }
                else {
                    // Normal route progression
                    next_track_0based = route[route_index];
                }
            }


            // Consume coal for movement (before trying to move)
            if (coal_level.load() < COAL_CONSUMPTION_PER_MOVE) {
                safe_print("[Train " + std::to_string(id) + "] Ran out of coal! Waiting for refill. Current level: " + std::to_string(coal_level.load()));
                if (current_track.load() != COAL_REFILL_STATION_ID) {
                    // If not at refill station, prioritize getting there
                    next_track_0based = COAL_REFILL_STATION_ID;
                }
                else {
                    // If at refill station, just wait for it to be free
                    refillCoal(); // Attempt to refill immediately if at station and out of coal
                    std::this_thread::sleep_for(std::chrono::milliseconds(TRAIN_ACTION_DELAY_MS));
                    continue; // Re-evaluate in next loop iteration
                }
            }
            else {
                coal_level -= COAL_CONSUMPTION_PER_MOVE;
            }


            // --- Phase 1: Attempt to reserve the next track ---
            bool reserved_next_track = false;
            auto start_wait_for_track = std::chrono::steady_clock::now();
            while (simulation_running.load() && !reserved_next_track) {
                std::unique_lock<std::mutex> lock_next_track(mtx_tracks[next_track_0based]);

                // Wait until the next track is free (not occupied AND not reserved by another train)
                // DECKIEL: Simplified wait condition. It should be free OR reserved by ME
                cv_tracks[next_track_0based].wait(lock_next_track, [this, next_track_0based] {
                    return (track_occupied_by_train[next_track_0based] == -1 && track_reserved_by_train[next_track_0based] == -1) ||
                        (track_reserved_by_train[next_track_0based] == id) || // It's reserved by me
                        !simulation_running.load();
                    });

                if (!simulation_running.load()) break; // Terminate if simulation ends

                // Now, if it's free or reserved by me, attempt to reserve it
                if (track_occupied_by_train[next_track_0based] == -1 && (track_reserved_by_train[next_track_0based] == -1 || track_reserved_by_train[next_track_0based] == id)) {
                    track_reserved_by_train[next_track_0based] = id;
                    safe_print("[Train " + std::to_string(id) + "] Reserved Station " + std::to_string(next_track_0based + 1) + ".");
                    reserved_next_track = true;
                }
                // If it's occupied by someone else, or reserved by someone else, we just wait again.
                // The wait condition should have prevented us from getting here if someone else has it.
                // If we get here and it's not free or reserved by us, it's a bug in wait condition or state.
            }
            if (!simulation_running.load()) break; // Exit if simulation ended during wait
            auto end_wait_for_track = std::chrono::steady_clock::now();
            track_wait_times_ms.push_back(std::chrono::duration_cast<std::chrono::milliseconds>(end_wait_for_track - start_wait_for_track).count());


            // Simulate the physical movement from old_current_track to next_track
            auto start_movement_time = std::chrono::steady_clock::now();
            std::this_thread::sleep_for(std::chrono::milliseconds(TRAIN_ACTION_DELAY_MS));

            // --- Phase 2: Acquire resources and update state ---
            // Lock old and new tracks in a consistent order (ascending ID) to prevent deadlocks
            std::vector<std::unique_lock<std::mutex>> locks;
            std::vector<int> tracks_to_lock;

            if (old_current_track_0based != -1) {
                tracks_to_lock.push_back(old_current_track_0based);
            }
            tracks_to_lock.push_back(next_track_0based);

            std::sort(tracks_to_lock.begin(), tracks_to_lock.end());
            tracks_to_lock.erase(std::unique(tracks_to_lock.begin(), tracks_to_lock.end()), tracks_to_lock.end());

            for (int track_id_to_lock : tracks_to_lock) {
                locks.emplace_back(std::unique_lock<std::mutex>(mtx_tracks[track_id_to_lock]));
            }

            // Now, with locks held, update the state:
            // 1. Release the old track
            if (old_current_track_0based != -1) {
                // Record end of occupancy for the old track
                {
                    std::lock_guard<std::mutex> lock_intervals(mtx_track_occupancy_intervals);
                    if (!track_occupancy_intervals[old_current_track_0based].empty()) {
                        track_occupancy_intervals[old_current_track_0based].back().second = std::chrono::steady_clock::now();
                    }
                }

                track_occupied_by_train[old_current_track_0based] = -1;
                track_reserved_by_train[old_current_track_0based] = -1; // Also clear reservation
                safe_print("[Train " + std::to_string(id) + "] Released Station " + std::to_string(old_current_track_0based + 1) + ".");
                cv_tracks[old_current_track_0based].notify_all(); // Notify all, as multiple trains might wait
                std::this_thread::sleep_for(std::chrono::milliseconds(TRAIN_ACTION_DELAY_MS / 2));
            }

            // 2. Occupy the new track
            // Fix: Change the condition to allow occupying if it's reserved by *this* train
            if (track_occupied_by_train[next_track_0based] == -1 &&
                (track_reserved_by_train[next_track_0based] == -1 || track_reserved_by_train[next_track_0based] == id)) {

                track_occupied_by_train[next_track_0based] = id;
                track_reserved_by_train[next_track_0based] = -1; // Clear reservation now that it's occupied
                current_track = next_track_0based; // Update atomic current_track (0-based)
                safe_print("[Train " + std::to_string(id) + "] Entered Station " + std::to_string(current_track.load() + 1) + ".");
                // Record start of occupancy for the new track
                {
                    std::lock_guard<std::mutex> lock_intervals(mtx_track_occupancy_intervals);
                    track_occupancy_intervals[next_track_0based].push_back({ std::chrono::steady_clock::now(), std::chrono::steady_clock::now() }); // End time updated later
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(TRAIN_ACTION_DELAY_MS));
            }
            else {
                safe_print("[Train " + std::to_string(id) + "] WARNING: Could not occupy Station " + std::to_string(next_track_0based + 1) + " as it was unexpectedly taken or still incorrectly reserved. Retrying.");
                // This scenario indicates a critical race condition if it happens after reservation and locking.
                // It's important to debug if this message appears frequently.
                std::this_thread::sleep_for(std::chrono::milliseconds(TRAIN_ACTION_DELAY_MS));
                continue; // Go back to the beginning of the while loop to re-attempt
            }


            // All locks are automatically released here by unique_lock destructors
            auto end_movement_time = std::chrono::steady_clock::now();
            movement_times_ms.push_back(std::chrono::duration_cast<std::chrono::milliseconds>(end_movement_time - start_movement_time).count());

            // --- Disembark passengers if at their destination ---
            // Now done AFTER occupying the track, ensuring current_track is up-to-date
            if (current_track.load() != -1) { // Only disembark if we just arrived at a station
                std::vector<std::string> disembarking_passengers;
                for (const auto& pair : passengers_on_board) {
                    const std::string& passenger_name = pair.first;
                    int dest_station = std::get<2>(pair.second);
                    if (dest_station == current_track.load()) { // Check if current station is destination
                        disembarking_passengers.push_back(passenger_name);
                    }
                }

                for (const auto& passenger_name : disembarking_passengers) {
                    auto it = passengers_on_board.find(passenger_name);
                    if (it != passengers_on_board.end()) {
                        int car_idx = std::get<0>(it->second);
                        int seat_idx = std::get<1>(it->second);
                        if (car_idx >= 0 && car_idx < cars.size() && cars[car_idx]) {
                            cars[car_idx]->releaseSeat(seat_idx);
                            safe_print("[Train " + std::to_string(id) + "] Passenger " + passenger_name + " disembarked at Station " + std::to_string(current_track.load() + 1) + " from car " + std::to_string(car_idx + 1) + ", seat " + std::to_string(seat_idx + 1) + ".");
                            std::this_thread::sleep_for(std::chrono::milliseconds(PASSENGER_ACTION_DELAY_MS / 4));
                        }
                        // Update passenger report entry for disembarkation
                        {
                            std::lock_guard<std::mutex> lock(mtx_passenger_reports);
                            for (auto& entry : passenger_reports) {
                                if (entry.passenger_name == passenger_name && entry.boarded_successfully && entry.disembark_time.time_since_epoch().count() == 0) {
                                    entry.disembark_time = std::chrono::steady_clock::now();
                                    // Check if this disembarkation was at the intended destination
                                    if (std::get<2>(it->second) == current_track.load()) {
                                        entry.reached_destination = true;
                                    }
                                    break;
                                }
                            }
                        }
                        passengers_on_board.erase(it); // Remove passenger from on-board list
                    }
                }
                if (!disembarking_passengers.empty()) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(TRAIN_ACTION_DELAY_MS));
                }
            }


            // --- Board passengers at the current station if available ---
            // Passengers only board if train is currently at a valid station AND not going to refill coal
            if (current_track.load() != -1 && coal_level.load() >= COAL_LOW_THRESHOLD) {
                std::vector<std::tuple<std::string, int, int>> boarding_passengers_candidates;
                {
                    std::unique_lock<std::mutex> lock(mtx_passengers_waiting);
                    if (passengers_waiting_for_train.count(id)) {
                        auto& waiting_list = passengers_waiting_for_train[id];

                        for (auto it = waiting_list.begin(); it != waiting_list.end(); ) {
                            int start_station = std::get<1>(*it);
                            if (start_station == current_track.load()) { // Use current_track for boarding station
                                boarding_passengers_candidates.push_back(std::move(*it)); // Move the tuple
                                it = waiting_list.erase(it); // Erase from original and get next iterator
                            }
                            else {
                                ++it; // Move to next passenger
                            }
                        }
                    }
                } // Unlock mtx_passengers_waiting

                for (const auto& passenger_info : boarding_passengers_candidates) {
                    const std::string& passenger_name = std::get<0>(passenger_info);
                    int dest_station = std::get<2>(passenger_info);
                    bool found_seat = false;
                    for (size_t w = 0; w < cars.size(); ++w) {
                        int occupied_seat_idx = -1;
                        if (cars[w]->occupySeat(occupied_seat_idx, passenger_name)) {
                            passengers_on_board[passenger_name] = { static_cast<int>(w), occupied_seat_idx, dest_station };
                            safe_print("[Train " + std::to_string(id) + "] Passenger " + passenger_name + " boarded at Station " + std::to_string(current_track.load() + 1) + " (To: " + std::to_string(dest_station + 1) + ") and occupied seat in car " + std::to_string(w + 1) + ", seat " + std::to_string(occupied_seat_idx + 1) + ".");
                            found_seat = true;
                            std::this_thread::sleep_for(std::chrono::milliseconds(PASSENGER_ACTION_DELAY_MS / 2));
                            // Update passenger report entry for boarding
                            {
                                std::lock_guard<std::mutex> lock(mtx_passenger_reports);
                                for (auto& entry : passenger_reports) {
                                    if (entry.passenger_name == passenger_name) {
                                        entry.board_time = std::chrono::steady_clock::now();
                                        entry.boarded_successfully = true;
                                        entry.had_to_rewait = false; // Cleared if they successfully board
                                        break;
                                    }
                                }
                            }
                            break;
                        }
                    }
                    if (!found_seat) {
                        safe_print("[Train " + std::to_string(id) + "] Passenger " + passenger_name + " could not find a free seat at Station " + std::to_string(current_track.load() + 1) + "! Re-adding to waiting list.");
                        // Re-add to waiting list if no seat was found
                        std::lock_guard<std::mutex> lock(mtx_passengers_waiting);
                        passengers_waiting_for_train[id].push_back(passenger_info);
                        cv_passengers_ready.notify_one(); // Notify potentially waiting passenger threads
                        // Mark passenger as having to re-wait
                        std::lock_guard<std::mutex> lock_report(mtx_passenger_reports);
                        for (auto& entry : passenger_reports) {
                            if (entry.passenger_name == passenger_name) {
                                entry.had_to_rewait = true;
                                break;
                            }
                        }
                    }
                }
                if (!boarding_passengers_candidates.empty()) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(TRAIN_ACTION_DELAY_MS));
                }
            }

            // Refill coal if at the refill station AND current coal is below max capacity
            if (current_track.load() == COAL_REFILL_STATION_ID && coal_level.load() < MAX_COAL_CAPACITY) {
                refillCoal();
            }

            // Simulate travel time on the track
            std::this_thread::sleep_for(std::chrono::milliseconds(rng() % 1000 + 500));

            // Move to the next point on the route ONLY if not currently at refill station and topped up
            if (current_track.load() != COAL_REFILL_STATION_ID || coal_level.load() >= MAX_COAL_CAPACITY) {
                route_index = (route_index + 1) % route.size();
                if (route_index == 0) { // Completed a full cycle of the route
                    full_route_cycles++;
                }
            }

            // At the end of the route loop, ensure all passengers are disembarked if any remain
            if (route_index == 0 && !passengers_on_board.empty() && current_track.load() != COAL_REFILL_STATION_ID) {
                safe_print("[Train " + std::to_string(id) + "] Completed route loop. Forcing disembarkation of remaining passengers.");
                std::vector<std::string> remaining_passengers_names;
                for (const auto& pair : passengers_on_board) {
                    remaining_passengers_names.push_back(pair.first);
                }
                for (const auto& passenger_name : remaining_passengers_names) {
                    auto it = passengers_on_board.find(passenger_name);
                    if (it != passengers_on_board.end()) {
                        int car_idx = std::get<0>(it->second);
                        int seat_idx = std::get<1>(it->second);
                        if (car_idx >= 0 && car_idx < cars.size() && cars[car_idx]) {
                            cars[car_idx]->releaseSeat(seat_idx);
                            safe_print("[Train " + std::to_string(id) + "] Passenger " + passenger_name + " forced disembarkation from car " + std::to_string(car_idx + 1) + ", seat " + std::to_string(seat_idx + 1) + ".");
                            std::this_thread::sleep_for(std::chrono::milliseconds(PASSENGER_ACTION_DELAY_MS / 4));
                        }
                        // Update passenger report entry for forced disembarkation
                        {
                            std::lock_guard<std::mutex> lock(mtx_passenger_reports);
                            for (auto& entry : passenger_reports) {
                                if (entry.passenger_name == passenger_name && entry.boarded_successfully && entry.disembark_time.time_since_epoch().count() == 0) {
                                    entry.disembark_time = std::chrono::steady_clock::now();
                                    // Even if forced, mark if they happened to be at their destination
                                    if (std::get<2>(it->second) == current_track.load()) {
                                        entry.reached_destination = true;
                                    }
                                    break;
                                }
                            }
                        }
                        passengers_on_board.erase(it);
                    }
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(TRAIN_ACTION_DELAY_MS * 2)); // Longer delay at terminal station
            }
        }

        // After simulation ends, ensure the train releases its last track
        if (current_track.load() != -1) {
            std::lock_guard<std::mutex> lock_last(mtx_tracks[current_track.load()]);
            // Record end of occupancy for the last track
            {
                std::lock_guard<std::mutex> lock_intervals(mtx_track_occupancy_intervals);
                if (!track_occupancy_intervals[current_track.load()].empty()) {
                    track_occupancy_intervals[current_track.load()].back().second = std::chrono::steady_clock::now();
                }
            }
            track_occupied_by_train[current_track.load()] = -1;
            track_reserved_by_train[current_track.load()] = -1; // Clear reservation
            safe_print("[Train " + std::to_string(id) + "] Released final Station " + std::to_string(current_track.load() + 1) + " at end of simulation.");
            cv_tracks[current_track.load()].notify_all();
            current_track = -1;
        }

        safe_print("[Train " + std::to_string(id) + "] Journey concluded.");
    }

private:
    std::string getRouteString() const {
        std::string s = "{";
        for (size_t i = 0; i < route.size(); ++i) {
            s += std::to_string(route[i] + 1); // Display 1-based station ID
            if (i < route.size() - 1) {
                s += ", ";
            }
        }
        s += "}";
        return s;
    }
};

class Passenger {
private:
    std::string name;
    int chosen_train_id; // 1-based ID
    int start_station;   // 0-based track ID
    int destination_station; // 0-based track ID
    std::mt19937 rng;

public:
    Passenger(std::string p_name) : name(std::move(p_name)) {
        rng.seed(std::chrono::system_clock::now().time_since_epoch().count() + std::hash<std::string>{}(name));
        chosen_train_id = -1; // No chosen train initially
        start_station = -1;
        destination_station = -1;

        // Record arrival at station time immediately
        std::lock_guard<std::mutex> lock(mtx_passenger_reports);
        passenger_reports.push_back({ name, std::chrono::steady_clock::now(), {}, {}, false, false, false });
    }

    void chooseTrainAndRoute(const std::vector<std::unique_ptr<Train>>& available_trains_uptr) {
        if (!simulation_running.load()) return;

        std::vector<Train*> available_trains;
        for (const auto& p_uptr : available_trains_uptr) {
            available_trains.push_back(p_uptr.get());
        }

        if (available_trains.empty()) {
            safe_print("[Passenger " + name + "] No available trains.");
            return;
        }

        // Try to find a suitable train and route
        bool chosen = false;
        int attempts = 0;
        const int max_attempts = 10; // Prevent infinite loops if no suitable route found

        while (!chosen && simulation_running.load() && attempts < max_attempts) {
            attempts++;
            std::uniform_int_distribution<> train_dist(0, static_cast<int>(available_trains.size() - 1));
            Train* candidate_train = available_trains[train_dist(rng)];

            const auto& train_route = candidate_train->getRoute();
            if (train_route.size() < 2) continue; // Route must have at least a start and end

            std::uniform_int_distribution<> station_dist(0, static_cast<int>(train_route.size() - 1));

            int s_idx = station_dist(rng);
            int d_idx = station_dist(rng);

            // Ensure start and destination are different
            while (s_idx == d_idx) {
                d_idx = station_dist(rng);
            }

            start_station = train_route[s_idx];
            destination_station = train_route[d_idx];
            chosen_train_id = candidate_train->getId();

            // Check if both start and destination stations are on the chosen train's route
            bool start_on_route = std::find(train_route.begin(), train_route.end(), start_station) != train_route.end();
            bool dest_on_route = std::find(train_route.begin(), train_route.end(), destination_station) != train_route.end();

            if (start_on_route && dest_on_route) {
                chosen = true;
            }
            else {
                continue; // Try another train/route combination
            }
        }

        if (chosen) {
            safe_print("[Passenger " + name + "] Chose Train " + std::to_string(chosen_train_id) +
                " from Station " + std::to_string(start_station + 1) +
                " to Station " + std::to_string(destination_station + 1) + ".");
            std::this_thread::sleep_for(std::chrono::milliseconds(PASSENGER_ACTION_DELAY_MS));

            {
                std::lock_guard<std::mutex> lock(mtx_passengers_waiting);
                passengers_waiting_for_train[chosen_train_id].emplace_back(name, start_station, destination_station);
            }
            cv_passengers_ready.notify_one();
        }
        else {
            safe_print("[Passenger " + name + "] Failed to choose a suitable train/route after " + std::to_string(max_attempts) + " attempts. Giving up.");
            // Mark passenger as not boarded if they give up
            std::lock_guard<std::mutex> lock(mtx_passenger_reports);
            for (auto& entry : passenger_reports) {
                if (entry.passenger_name == name) {
                    entry.boarded_successfully = false;
                    entry.reached_destination = false; // Cannot reach if not boarded
                    break;
                }
            }
        }
    }
};

// --- Function for drawing the railway map (visualization) ---
void draw_railway_map(const std::vector<std::unique_ptr<Train>>& trains,
    const std::vector<int>& track_occupied_status, // This is track_occupied_by_train
    const std::vector<int>& track_reserved_status) // This is track_reserved_by_train
{
    std::lock_guard<std::mutex> lock(mtx_cout); // Protect the console

    // On Windows (requires #include <cstdlib> and might be slower):
    // system("cls"); // Uncomment if running on Windows and console clear is desired
    // On Linux/macOS:
    std::cout << "\033[2J\033[H"; // Clear screen and set cursor to home

    std::cout << "--- RAILWAY MAP ---" << std::endl;
    std::cout << "---------------------" << std::endl;

    // Station Status Section
    std::cout << "--- STATIONS STATUS ---" << std::endl;
    std::cout << "Station ID | Status            | Occupied By | Reserved By" << std::endl;
    std::cout << "----------+-------------------+-------------+------------" << std::endl;

    for (size_t i = 0; i < track_occupied_status.size(); ++i) {
        // Display 1-based station ID
        std::cout << std::setw(9) << i + 1 << " | ";

        // Determine status string
        std::string status_str;
        if (track_occupied_status[i] != -1) {
            status_str = "[== OCCUPIED ==]";
        }
        else if (track_reserved_status[i] != -1) {
            status_str = "[-- RESERVED --]";
        }
        else {
            status_str = "[-- FREE -----]";
        }
        std::cout << status_str << " | ";

        // Display occupied by
        if (track_occupied_status[i] != -1) {
            std::cout << "Train " << std::setw(2) << track_occupied_status[i];
        }
        else {
            std::cout << std::setw(9) << "None";
        }
        std::cout << " | ";

        // Display reserved by
        if (track_reserved_status[i] != -1) {
            std::cout << "Train " << std::setw(2) << track_reserved_status[i];
        }
        else {
            std::cout << std::setw(9) << "None";
        }
        std::cout << std::endl;
    }
    std::cout << "---------------------" << std::endl;


    std::cout << "--- TRAIN STATUS ---" << std::endl;
    std::cout << "Train Status (In Depot means train is not on a station):" << std::endl;
    for (const auto& p_uptr : trains) {
        std::string route_str = "{";
        const auto& route = p_uptr->getRoute(); // Route elements are 0-based track IDs
        for (size_t i = 0; i < route.size(); ++i) {
            route_str += std::to_string(route[i] + 1); // Display 1-based station ID
            if (i < route.size() - 1) {
                route_str += ",";
            }
        }
        route_str += "}";

        // Display 1-based Train ID and 1-based current Station ID
        std::cout << "  Train " << std::setw(2) << p_uptr->getId()
            << " at: " << (p_uptr->getCurrentTrack() != -1 ? "Station " + std::to_string(p_uptr->getCurrentTrack() + 1) : "In Depot")
            << " | Route: " << route_str
            << " | Coal: " << std::setw(3) << p_uptr->getCoalLevel() << "/" << MAX_COAL_CAPACITY << std::endl;

        // Display passenger locations in each car
        std::cout << "    Passengers onboard:" << std::endl;
        const auto& cars_on_train = p_uptr->getCars();
        for (size_t c = 0; c < cars_on_train.size(); ++c) {
            // Display 1-based Car ID
            std::string car_occupancy = "      Car " + std::to_string(c + 1) + ": [";
            for (int s = 0; s < SEATS_PER_CAR; ++s) {
                std::string passenger_name = cars_on_train[c]->getPassengerAtSeat(s);
                if (!passenger_name.empty()) {
                    // Extract just the number from "Pas_X"
                    size_t underscore_pos = passenger_name.find('_');
                    std::string passenger_number_str = "";
                    if (underscore_pos != std::string::npos) {
                        passenger_number_str = passenger_name.substr(underscore_pos + 1);
                    }
                    else {
                        passenger_number_str = passenger_name; // Fallback if format is unexpected
                    }

                    car_occupancy += "Pas(" + passenger_number_str + ")";
                }
                else {
                    car_occupancy += "Free(0)"; // Display Free(0) for empty seats
                }
                if (s < SEATS_PER_CAR - 1) {
                    car_occupancy += ", ";
                }
            }
            car_occupancy += "]";
            // Always print car details, regardless of passengers
            std::cout << car_occupancy << std::endl;
        }
    }
    std::cout << "---------------------" << std::endl << std::endl;
}

// Thread responsible for drawing
void visualization_thread_func(const std::vector<std::unique_ptr<Train>>& trains,
    const std::vector<int>& track_occupied_status,
    const std::vector<int>& track_reserved_status) {
    while (simulation_running.load()) {
        draw_railway_map(trains, track_occupied_status, track_reserved_status);
        std::this_thread::sleep_for(std::chrono::milliseconds(VISUALIZATION_REFRESH_MS)); // Refresh every VISUALIZATION_REFRESH_MS
    }
    // Draw the last frame before exiting
    draw_railway_map(trains, track_occupied_status, track_reserved_status);
}


// Function to generate a random route for a train
// Generates 0-based track IDs
std::vector<int> generateRandomRoute(int num_tracks, int min_length, int max_length, std::mt19937& rng) {
    std::vector<int> route;
    if (num_tracks < min_length) {
        // Not enough tracks to form a route of minimum length.
        std::cerr << "Warning: Not enough tracks (" << num_tracks << ") to generate a route of minimum length (" << min_length << ")." << std::endl;
        return { 0 }; // Default to track 0 if impossible to generate a proper route
    }

    std::uniform_int_distribution<> length_dist(min_length, std::min(max_length, num_tracks));
    int route_length = length_dist(rng);

    // Create a vector of all possible track IDs (0-based)
    std::vector<int> all_tracks(num_tracks);
    std::iota(all_tracks.begin(), all_tracks.end(), 0); // Fills with 0, 1, 2, ..., num_tracks-1

    // Shuffle the tracks to get unique random ones for the route
    std::shuffle(all_tracks.begin(), all_tracks.end(), rng);

    // Take the first 'route_length' unique tracks
    for (int i = 0; i < route_length; ++i) {
        route.push_back(all_tracks[i]);
    }
    return route;
}

// --- Reporting Function ---
void generate_final_report(const std::vector<std::unique_ptr<Train>>& trains, std::chrono::steady_clock::time_point simulation_start_time) {
    std::lock_guard<std::mutex> lock_cout(mtx_cout);
    std::cout << "\n\n--- SIMULATION REPORT ---" << std::endl;
    std::cout << "---------------------------\n" << std::endl;

    // Passenger Statistics
    long long total_passenger_wait_time_ms = 0;
    long long total_passenger_travel_time_ms = 0;
    int boarded_passengers_count = 0;
    int unboarded_passengers_count = 0;
    int reached_destination_passengers_count = 0;
    int had_to_rewait_passengers_count = 0;
    long long min_wait_time_ms = -1;
    long long max_wait_time_ms = 0;
    long long min_travel_time_ms = -1;
    long long max_travel_time_ms = 0;

    for (const auto& entry : passenger_reports) {
        if (entry.boarded_successfully) {
            boarded_passengers_count++;
            if (entry.board_time.time_since_epoch().count() != 0) {
                long long wait_time = std::chrono::duration_cast<std::chrono::milliseconds>(entry.board_time - entry.arrival_at_station_time).count();
                total_passenger_wait_time_ms += wait_time;
                if (min_wait_time_ms == -1 || wait_time < min_wait_time_ms) min_wait_time_ms = wait_time;
                if (wait_time > max_wait_time_ms) max_wait_time_ms = wait_time;
            }
            if (entry.disembark_time.time_since_epoch().count() != 0) {
                long long travel_time = std::chrono::duration_cast<std::chrono::milliseconds>(entry.disembark_time - entry.board_time).count();
                total_passenger_travel_time_ms += travel_time;
                if (min_travel_time_ms == -1 || travel_time < min_travel_time_ms) min_travel_time_ms = travel_time;
                if (travel_time > max_travel_time_ms) max_travel_time_ms = travel_time;
            }
        }
        else {
            unboarded_passengers_count++;
        }

        if (entry.reached_destination) {
            reached_destination_passengers_count++;
        }
        if (entry.had_to_rewait) {
            had_to_rewait_passengers_count++;
        }
    }

    std::cout << "--- Passenger Statistics ---" << std::endl;
    std::cout << "Total passengers created: " << passenger_reports.size() << std::endl;
    std::cout << "Passengers who boarded successfully: " << boarded_passengers_count << std::endl;
    std::cout << "Passengers who reached destination: " << reached_destination_passengers_count << std::endl;
    std::cout << "Passengers who did NOT board: " << unboarded_passengers_count << std::endl;
    std::cout << "Passengers who had to re-wait (no seat found): " << had_to_rewait_passengers_count << std::endl;

    if (boarded_passengers_count > 0) {
        std::cout << "Average passenger waiting time on station: "
            << std::fixed << std::setprecision(2) << static_cast<double>(total_passenger_wait_time_ms) / boarded_passengers_count / 1000.0
            << " seconds" << std::endl;
        std::cout << "Min passenger waiting time on station: " << static_cast<double>(min_wait_time_ms) / 1000.0 << " seconds" << std::endl;
        std::cout << "Max passenger waiting time on station: " << static_cast<double>(max_wait_time_ms) / 1000.0 << " seconds" << std::endl;

        std::cout << "Average passenger travel time (onboard): "
            << std::fixed << std::setprecision(2) << static_cast<double>(total_passenger_travel_time_ms) / boarded_passengers_count / 1000.0
            << " seconds" << std::endl;
        std::cout << "Min passenger travel time (onboard): " << static_cast<double>(min_travel_time_ms) / 1000.0 << " seconds" << std::endl;
        std::cout << "Max passenger travel time (onboard): " << static_cast<double>(max_travel_time_ms) / 1000.0 << " seconds" << std::endl;
    }
    else {
        std::cout << "No passengers successfully boarded to calculate average travel times." << std::endl;
    }
    std::cout << std::endl;

    // Train Statistics
    std::cout << "--- Train Statistics ---" << std::endl;
    long long total_coal_consumed_all_trains = 0;
    for (const auto& train_ptr : trains) {
        long long total_track_wait_time = 0;
        for (long long wait_time : train_ptr->getTrackWaitTimes()) {
            total_track_wait_time += wait_time;
        }
        double avg_track_wait_time = train_ptr->getTrackWaitTimes().empty() ? 0.0 :
            static_cast<double>(total_track_wait_time) / train_ptr->getTrackWaitTimes().size() / 1000.0;

        long long total_coal_refill_time = 0;
        for (long long refill_time : train_ptr->getCoalRefillTimes()) {
            total_coal_refill_time += refill_time;
        }
        double avg_coal_refill_time = train_ptr->getCoalRefillTimes().empty() ? 0.0 :
            static_cast<double>(total_coal_refill_time) / train_ptr->getCoalRefillTimes().size() / 1000.0;

        long long total_movement_time = 0;
        for (long long move_time : train_ptr->getMovementTimes()) {
            total_movement_time += move_time;
        }

        long long total_train_active_time_ms = total_track_wait_time + total_coal_refill_time + total_movement_time;
        double percent_movement = 0.0;
        double percent_track_wait = 0.0;
        double percent_coal_refill = 0.0;

        if (total_train_active_time_ms > 0) {
            percent_movement = static_cast<double>(total_movement_time) / total_train_active_time_ms * 100.0;
            percent_track_wait = static_cast<double>(total_track_wait_time) / total_train_active_time_ms * 100.0;
            percent_coal_refill = static_cast<double>(total_coal_refill_time) / total_train_active_time_ms * 100.0;
        }

        std::cout << "Train " << train_ptr->getId() << ":" << std::endl;
        std::cout << "  Number of cars: " << train_ptr->getCars().size() << std::endl;
        std::cout << "  Full route cycles completed: " << train_ptr->getFullRouteCycles() << std::endl;
        std::cout << "  Average track waiting time: " << std::fixed << std::setprecision(2) << avg_track_wait_time << " seconds" << std::endl;
        std::cout << "  Total track waiting time: " << std::fixed << std::setprecision(2) << static_cast<double>(total_track_wait_time) / 1000.0 << " seconds" << std::endl;
        std::cout << "  Average coal refill time: " << std::fixed << std::setprecision(2) << avg_coal_refill_time << " seconds" << std::endl;
        std::cout << "  Total coal refill time: " << std::fixed << std::setprecision(2) << static_cast<double>(total_coal_refill_time) / 1000.0 << " seconds" << std::endl;
        std::cout << "  Total movement time: " << std::fixed << std::setprecision(2) << static_cast<double>(total_movement_time) / 1000.0 << " seconds" << std::endl;
        std::cout << "  Time distribution:" << std::endl;
        std::cout << "    - Moving: " << std::fixed << std::setprecision(2) << percent_movement << "%" << std::endl;
        std::cout << "    - Waiting for track: " << std::fixed << std::setprecision(2) << percent_track_wait << "%" << std::endl;
        std::cout << "    - Refilling coal: " << std::fixed << std::setprecision(2) << percent_coal_refill << "%" << std::endl;
        std::cout << std::endl;

        total_coal_consumed_all_trains += (MAX_COAL_CAPACITY - train_ptr->getCoalLevel()) + (train_ptr->getCoalRefillTimes().size() * COAL_REFILL_AMOUNT); // Simplified coal consumption
    }
    std::cout << std::endl;


    // Infrastructure Statistics (Track Utilization)
    std::cout << "--- Infrastructure Statistics ---" << std::endl;
    long long total_simulation_duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - simulation_start_time).count();

    for (size_t i = 0; i < NUM_TRACKS; ++i) {
        long long total_occupancy_time_ms = 0;
        long long total_wait_at_coal_station = 0;

        {
            std::lock_guard<std::mutex> lock_intervals(mtx_track_occupancy_intervals);
            for (const auto& interval : track_occupancy_intervals[i]) {
                if (interval.second.time_since_epoch().count() != 0) { // Ensure end time is recorded
                    total_occupancy_time_ms += std::chrono::duration_cast<std::chrono::milliseconds>(interval.second - interval.first).count();
                }
                else {
                    // If simulation ended, use current time as end time for ongoing occupancies
                    total_occupancy_time_ms += std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - interval.first).count();
                }
            }
        }

        double track_utilization_percent = 0.0;
        if (total_simulation_duration_ms > 0) {
            track_utilization_percent = (static_cast<double>(total_occupancy_time_ms) / total_simulation_duration_ms) * 100.0;
        }

        std::cout << "Track " << i + 1 << " Utilization: " << std::fixed << std::setprecision(2) << track_utilization_percent << "%" << std::endl;

        // Specific statistics for the coal refill station
        if (i == COAL_REFILL_STATION_ID) {
            // Calculate average waiting time at the coal station from train reports directly
            for (const auto& train_ptr : trains) {
                for (long long refill_time : train_ptr->getCoalRefillTimes()) {
                    total_wait_at_coal_station += refill_time;
                }
            }
            double avg_coal_station_wait = trains.empty() ? 0.0 : static_cast<double>(total_wait_at_coal_station) / trains.size() / 1000.0; // Average across trains

            std::cout << "  (Coal Refill Station) Average wait time for all trains: " << std::fixed << std::setprecision(2) << avg_coal_station_wait << " seconds" << std::endl;
        }
    }
    std::cout << std::endl;

    // Overall Statistics
    std::cout << "--- Overall Statistics ---" << std::endl;
    std::cout << "Total passengers delivered to destination: " << reached_destination_passengers_count << std::endl;
    std::cout << "Total coal consumed across all trains: " << total_coal_consumed_all_trains << std::endl;
    std::cout << "Total simulation time: " << std::fixed << std::setprecision(2) << static_cast<double>(total_simulation_duration_ms) / 1000.0 << " seconds" << std::endl;


    std::cout << "---------------------------\n" << std::endl;
}


// --- Main Simulation Function ---
int main() {
    std::cout << "--- Railway Simulation with Thread Conflicts and Visualization ---" << std::endl;

    // Initialize tracks (0-based internally, -1 if free)
    track_occupied_by_train.resize(NUM_TRACKS, -1); // All tracks are free initially
    track_reserved_by_train.resize(NUM_TRACKS, -1); // No tracks are reserved initially

    // Initialize track occupancy intervals
    track_occupancy_intervals.resize(NUM_TRACKS);


    std::vector<std::unique_ptr<Train>> trains_uptr;
    std::vector<std::thread> train_threads;

    // Initialize a random number generator for routes and car count
    std::mt19937 main_rng(std::chrono::system_clock::now().time_since_epoch().count());

    // Create trains with their random routes and random number of cars
    for (int i = 0; i < NUM_TRAINS; ++i) {
        std::uniform_int_distribution<> num_cars_dist(MIN_CARS_PER_TRAIN, MAX_CARS_PER_TRAIN);
        int num_cars_for_this_train = num_cars_dist(main_rng);

        // Generate a random route for each train (0-based track IDs)
        std::vector<int> random_route = generateRandomRoute(NUM_TRACKS, MIN_ROUTE_LENGTH, MAX_ROUTE_LENGTH, main_rng);
        if (random_route.empty()) {
            std::cerr << "Warning: Generated an empty route for Train " << i + 1 << ". Using default route." << std::endl;
            random_route = { 0 }; // Fallback to a default route to avoid errors
        }
        trains_uptr.push_back(std::make_unique<Train>(i + 1, random_route, num_cars_for_this_train)); // Assign 1-based Train ID
    }

    auto simulation_start_time = std::chrono::steady_clock::now();

    // Start the visualization thread first
    std::thread visualization_thread(visualization_thread_func, std::cref(trains_uptr), std::cref(track_occupied_by_train), std::cref(track_reserved_by_train));

    // Create threads for trains
    for (const auto& p_uptr : trains_uptr) {
        train_threads.emplace_back(&Train::run, p_uptr.get());
    }

    // Create passengers and assign them to trains with specific start/destination stations
    std::vector<std::unique_ptr<Passenger>> passengers_uptr;
    std::vector<std::thread> passenger_threads;
    // Example: fewer passengers than seats, but enough to create contention
    // We will generate more passengers to account for some failing to find a suitable route
    int total_possible_seats = NUM_TRAINS * MAX_CARS_PER_TRAIN * SEATS_PER_CAR;
    for (int i = 0; i < total_possible_seats * 2; ++i) { // Doubled for more traffic
        passengers_uptr.push_back(std::make_unique<Passenger>("Pas_" + std::to_string(i + 1))); // Assign 1-based Passenger ID
        passenger_threads.emplace_back(&Passenger::chooseTrainAndRoute, passengers_uptr.back().get(), std::cref(trains_uptr));
        // No sleep here, to generate passengers as fast as possible
        // std::this_thread::sleep_for(std::chrono::milliseconds(PASSENGER_ACTION_DELAY_MS));
    }

    // Wait for the specified simulation time
    std::this_thread::sleep_for(std::chrono::seconds(SIMULATION_TIME_SECONDS));

    // Signal threads to stop
    simulation_running = false;
    cv_passengers_ready.notify_all(); // Wake up passenger/train threads waiting for passengers
    for (size_t i = 0; i < NUM_TRACKS; ++i) {
        cv_tracks[i].notify_all(); // Wake up train threads waiting for tracks
    }
    cv_coal_station_free.notify_all(); // Wake up any train waiting for coal station

    // Wait for train threads to finish
    for (std::thread& t : train_threads) {
        if (t.joinable()) {
            t.join();
        }
    }

    // Wait for passenger threads to finish
    for (std::thread& t : passenger_threads) {
        if (t.joinable()) {
            t.join();
        }
    }

    // Wait for visualization thread to finish
    if (visualization_thread.joinable()) {
        visualization_thread.join();
    }

    // Generate and display the final report
    generate_final_report(trains_uptr, simulation_start_time);

    std::cout << "--- Simulation Finished ---" << std::endl;

    return 0;
}
