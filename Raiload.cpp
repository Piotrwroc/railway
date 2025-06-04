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
#include <memory>        // For std::unique_ptr
#include <functional>    // For std::ref
#include <atomic>        // For std::atomic<bool> to safely stop threads
#include <iomanip>       // For std::setw
#include <numeric>       // For std::iota
#include <algorithm>     // For std::shuffle, std::find, std::min, std::max
#include <cstdlib>       // For system("cls")
#include <tuple>         // For std::tuple

// --- Constants and Configuration ---
const int NUM_TRAINS = 5;
const int SEATS_PER_CAR = 2; // Reduced for more seat contention
const int NUM_CARS = 2; // Reduced for more seat contention
const int NUM_TRACKS = 5; // Keep moderate to allow for track contention without deadlocks
const int SIMULATION_TIME_SECONDS = 90; // Increased to allow more time for conflicts to develop and resolve
const int VISUALIZATION_REFRESH_MS = 250; // Faster refresh for more dynamic visualization
const int TRAIN_ACTION_DELAY_MS = 500; // Reduced to make trains move faster and increase track contention
const int PASSENGER_ACTION_DELAY_MS = 10; // Drastically reduced for faster passenger boarding attempts

// Define min/max length for random routes
const int MIN_ROUTE_LENGTH = 3; // Routes should be longer to increase track overlap
const int MAX_ROUTE_LENGTH = NUM_TRACKS; // Max route length means trains will traverse most tracks


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

    // Explicitly delete copy constructor and copy assignment operator
    // to ensure Car is never copied.
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

// Flaga do bezpiecznego zatrzymywania w¹tków
std::atomic<bool> simulation_running(true);


// Funkcja do bezpiecznego wypisywania na konsolê
void safe_print(const std::string& message) {
    std::lock_guard<std::mutex> lock(mtx_cout);
    std::cout << message << std::endl;
}

// --- Klasy reprezentuj¹ce elementy symulacji ---

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

public:
    Train(int p_id, const std::vector<int>& p_route)
        : id(p_id), current_track(-1), route(p_route), route_index(0) {
        for (int i = 0; i < NUM_CARS; ++i) {
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

    // Access to cars for visualization
    const std::vector<std::unique_ptr<Car>>& getCars() const {
        return cars;
    }


    // Funkcja do symulacji ruchu poci¹gu i obs³ugi konfliktów
    void run() {
        // Output Train ID as 1-based
        safe_print("[Train " + std::to_string(id) + "] Starting! My route: " + getRouteString());
        std::this_thread::sleep_for(std::chrono::milliseconds(TRAIN_ACTION_DELAY_MS)); // Spowolnienie

        while (simulation_running.load()) {
            int next_track_0based = route[route_index];
            int old_current_track_0based = current_track.load(); // 0-based

            // --- Disembark passengers if at their destination ---
            if (old_current_track_0based != -1) { // Only disembark if we just arrived at a station
                std::vector<std::string> disembarking_passengers;
                for (const auto& pair : passengers_on_board) {
                    const std::string& passenger_name = pair.first;
                    int dest_station = std::get<2>(pair.second);
                    if (dest_station == old_current_track_0based) { // Check if current station is destination
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
                            safe_print("[Train " + std::to_string(id) + "] Passenger " + passenger_name + " disembarked at Station " + std::to_string(old_current_track_0based + 1) + " from car " + std::to_string(car_idx + 1) + ", seat " + std::to_string(seat_idx + 1) + ".");
                            std::this_thread::sleep_for(std::chrono::milliseconds(PASSENGER_ACTION_DELAY_MS / 4));
                        }
                        passengers_on_board.erase(it); // Remove passenger from on-board list
                    }
                }
                if (!disembarking_passengers.empty()) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(TRAIN_ACTION_DELAY_MS));
                }
            }


            // --- Board passengers at the current station if available ---
            // This now happens at whatever station the train is currently at
            // after the disembarkation (which means, `old_current_track_0based` holds the current station)
            if (old_current_track_0based != -1) {
                std::vector<std::tuple<std::string, int, int>> boarding_passengers_candidates;
                {
                    std::unique_lock<std::mutex> lock(mtx_passengers_waiting);
                    if (passengers_waiting_for_train.count(id)) {
                        auto& waiting_list = passengers_waiting_for_train[id];
                        // Use a temporary vector for passengers to board this round
                        std::vector<std::tuple<std::string, int, int>> temp_waiting_list;

                        // Iterate through the waiting list and move eligible passengers to boarding_passengers_candidates
                        for (auto it = waiting_list.begin(); it != waiting_list.end(); ) {
                            int start_station = std::get<1>(*it);
                            if (start_station == old_current_track_0based) {
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
                            safe_print("[Train " + std::to_string(id) + "] Passenger " + passenger_name + " boarded at Station " + std::to_string(old_current_track_0based + 1) + " (To: " + std::to_string(dest_station + 1) + ") and occupied seat in car " + std::to_string(w + 1) + ", seat " + std::to_string(occupied_seat_idx + 1) + ".");
                            found_seat = true;
                            std::this_thread::sleep_for(std::chrono::milliseconds(PASSENGER_ACTION_DELAY_MS / 2));
                            break;
                        }
                    }
                    if (!found_seat) {
                        safe_print("[Train " + std::to_string(id) + "] Passenger " + passenger_name + " could not find a free seat at Station " + std::to_string(old_current_track_0based + 1) + "! Re-adding to waiting list.");
                        // Re-add to waiting list if no seat was found
                        std::lock_guard<std::mutex> lock(mtx_passengers_waiting);
                        passengers_waiting_for_train[id].push_back(passenger_info);
                        cv_passengers_ready.notify_one(); // Notify potentially waiting passenger threads
                    }
                }
                if (!boarding_passengers_candidates.empty()) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(TRAIN_ACTION_DELAY_MS));
                }
            }


            // --- Conflict 3 Simulation: Signaling and track blocks (Deadlock Prevention) ---
            if (route.empty()) {
                safe_print("[Train " + std::to_string(id) + "] No route defined, stopping.");
                break;
            }

            // --- Phase 1: Attempt to reserve the next track ---
            bool reserved_next_track = false;
            while (simulation_running.load() && !reserved_next_track) {
                std::unique_lock<std::mutex> lock_next_track(mtx_tracks[next_track_0based]);

                // Wait until the next track is free or already reserved by this train
                cv_tracks[next_track_0based].wait(lock_next_track, [this, next_track_0based] {
                    return track_occupied_by_train[next_track_0based] == -1 || // Free
                        track_reserved_by_train[next_track_0based] == id || // Reserved by me
                        !simulation_running.load();
                    });

                if (!simulation_running.load()) break; // Terminate if simulation ends

                if (track_occupied_by_train[next_track_0based] == -1) { // If truly free
                    track_reserved_by_train[next_track_0based] = id;
                    safe_print("[Train " + std::to_string(id) + "] Reserved Station " + std::to_string(next_track_0based + 1) + ".");
                    reserved_next_track = true;
                }
                else if (track_reserved_by_train[next_track_0based] == id) { // Already reserved by me, that's fine
                    safe_print("[Train " + std::to_string(id) + "] Station " + std::to_string(next_track_0based + 1) + " was already reserved by me.");
                    reserved_next_track = true;
                }
            }
            if (!simulation_running.load()) break; // Exit if simulation ended during wait

            // Simulate the physical movement from old_current_track to next_track
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
                locks.emplace_back(mtx_tracks[track_id_to_lock]);
            }

            // Now, with locks held, update the state:
            // 1. Release the old track
            if (old_current_track_0based != -1) {
                track_occupied_by_train[old_current_track_0based] = -1;
                track_reserved_by_train[old_current_track_0based] = -1; // Also clear reservation
                safe_print("[Train " + std::to_string(id) + "] Released Station " + std::to_string(old_current_track_0based + 1) + ".");
                cv_tracks[old_current_track_0based].notify_all(); // Notify all, as multiple trains might wait
                std::this_thread::sleep_for(std::chrono::milliseconds(TRAIN_ACTION_DELAY_MS / 2));
            }

            // 2. Occupy the new track
            track_occupied_by_train[next_track_0based] = id;
            track_reserved_by_train[next_track_0based] = -1; // Clear reservation now that it's occupied
            current_track = next_track_0based; // Update atomic current_track (0-based)
            safe_print("[Train " + std::to_string(id) + "] Entered Station " + std::to_string(current_track.load() + 1) + ".");
            std::this_thread::sleep_for(std::chrono::milliseconds(TRAIN_ACTION_DELAY_MS));

            // All locks are automatically released here by unique_lock destructors

            // Simulate travel time on the track
            std::this_thread::sleep_for(std::chrono::milliseconds(rng() % 1000 + 500));

            // Move to the next point on the route
            route_index = (route_index + 1) % route.size();

            // At the end of the route loop, ensure all passengers are disembarked if any remain
            if (route_index == 0 && !passengers_on_board.empty()) {
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
                        passengers_on_board.erase(it);
                    }
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(TRAIN_ACTION_DELAY_MS * 2)); // Longer delay at terminal station
            }
        }

        // After simulation ends, ensure the train releases its last track
        if (current_track.load() != -1) {
            std::lock_guard<std::mutex> lock_last(mtx_tracks[current_track.load()]);
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
            << " | Route: " << route_str << std::endl;

        // Display passenger locations in each car
        std::cout << "    Passengers onboard:" << std::endl;
        const auto& cars_on_train = p_uptr->getCars();
        // Removed the check for 'any_passengers_in_car' and the ' (None)' output
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

                    car_occupancy += "Passenger(" + passenger_number_str + ")";
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


// --- Main Simulation Function ---
int main() {
    std::cout << "--- Railway Simulation with Thread Conflicts and Visualization ---" << std::endl;

    // Initialize tracks (0-based internally, -1 if free)
    track_occupied_by_train.resize(NUM_TRACKS, -1); // All tracks are free initially
    track_reserved_by_train.resize(NUM_TRACKS, -1); // No tracks are reserved initially

    std::vector<std::unique_ptr<Train>> trains_uptr;
    std::vector<std::thread> train_threads;

    // Initialize a random number generator for routes
    std::mt19937 route_rng(std::chrono::system_clock::now().time_since_epoch().count());

    // Create trains with their random routes
    for (int i = 0; i < NUM_TRAINS; ++i) {
        // Generate a random route for each train (0-based track IDs)
        std::vector<int> random_route = generateRandomRoute(NUM_TRACKS, MIN_ROUTE_LENGTH, MAX_ROUTE_LENGTH, route_rng);
        if (random_route.empty()) {
            std::cerr << "Error: Generated an empty route for Train " << i + 1 << ". Using default route." << std::endl;
            random_route = { 0 }; // Fallback to a default route to avoid errors
        }
        trains_uptr.push_back(std::make_unique<Train>(i + 1, random_route)); // Assign 1-based Train ID
    }

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
    for (int i = 0; i < NUM_TRAINS * NUM_CARS * SEATS_PER_CAR * 2; ++i) { // Doubled for more traffic
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

    std::cout << "--- Simulation Finished ---" << std::endl;

    return 0;
}