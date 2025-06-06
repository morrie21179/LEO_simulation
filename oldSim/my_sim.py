import random
import math
import numpy as np
def generate_user_requests(num_users, num_views):
    reqs = []
    for uid in range(1, num_users + 1):
        lat = random.uniform(-90, 90)
        lon = random.uniform(-180, 180)
        view = random.randint(1, num_views)
        reqs.append((uid, lat, lon, view))
    return reqs

def area_model(satellite_positions, earth_radius=6371, psi=0.3):
    """
    Calculate the area covered by satellites.
    
    Parameters:
    - satellite_positions: List of (longitude, latitude) tuples for each satellite
    - earth_radius: Radius of Earth in km
    - psi: Half cone angle between covered area and Earth's core
    
    Returns:
    - Coverage data for each satellite and total coverage
    """
    coverage_data = {}
    
    for n, (lon, lat) in enumerate(satellite_positions, 1):
        # Convert to radians
        phi_n = math.radians(lon)  # The longitude of the satellite n
        theta_n = abs(math.radians(90 - lat))
        
        # Store satellite coverage information
        coverage_data[n] = {
            'position': (lon, lat),
            'phi_n': phi_n,
            'theta_n': theta_n,
            'psi': psi,
            'cos_psi': math.cos(psi)
        }
    
    return coverage_data

def coverage_model(height, theta_min, earth_radius=6371):
    """
    Calculate satellite coverage based on height and minimum elevation angle.
    
    Parameters:
    - height: Height of satellite above Earth's surface in km
    - theta_min: Minimum elevation angle in radians
    - earth_radius: Radius of Earth in km
    
    Returns:
    - psi: Coverage angle in radians
    - coverage_area: Surface area covered in square km
    """
    # Calculate psi according to the formula
    psi = math.acos((earth_radius / (earth_radius + height)) * 
                   math.cos(theta_min)) - theta_min
    
    # Calculate coverage area
    coverage_area = 2 * math.pi * (earth_radius ** 2) * (1 - math.cos(psi))
    
    return {
        'psi': psi,
        'coverage_area': coverage_area
    }

def is_point_covered(lat, lon, satellite_data):
    """
    Check if a point is covered by a satellite.
    
    Parameters:
    - lat, lon: Point coordinates in degrees
    - satellite_data: Satellite coverage data from area_model
    
    Returns:
    - True if covered, False otherwise
    """
    # Convert to radians
    phi = math.radians(lon)
    theta = math.radians(90 - lat)
    
    for sat_id, sat in satellite_data.items():
        phi_n = sat['phi_n']
        theta_n = sat['theta_n']
        cos_psi = sat['cos_psi']
        
        # Check if point is in coverage area
        if (math.sin(theta) * math.sin(theta_n) * math.cos(phi - phi_n) + 
            math.cos(theta) * math.cos(theta_n)) <= cos_psi:
            return True
    
    return False

def LRU(satellites, current_time, update_position_func):
    """
    1. Function to evolve satellite system to the next time step using LRU cache replacement.
    2. Parameters:
        - satellites: List of SatelliteWithCache objects
        - current_time: Current time step
        - update_position_func: Function to update satellite positions
    3. Returns:
        - Next time step
    """
    next_time = current_time + 1
    
    # Update positions based on orbital mechanics
    new_positions = update_position_func(next_time)
    
    for sat, new_pos in zip(satellites, new_positions):
        # Update position
        sat.update_position(next_time, new_pos)
        
        # Get current cache state
        current_cache = sat.get_cache(current_time)
        
        # Keep track of usage history (for LRU)
        if not hasattr(sat, 'lru_history'):
            sat.lru_history = list(current_cache)
        
        # Determine remaining views that could be added to cache
        remaining_views = universal_views - current_cache
        
        # Decide how many new views to fetch (random number between 0 and 3)
        num_new_views = random.randint(0, min(3, len(remaining_views)))
        
        if num_new_views > 0 and remaining_views:
            # Select random new views to add
            new_views = set(random.sample(list(remaining_views), num_new_views))
            # Initialize cost counters for this satellite
            if not hasattr(sat, 'cost_stats'):
                sat.cost_stats = {
                    'transmission_cost': 0,  # c_s = 2
                    'cache_miss_cost': 0,    # c_m = 10
                    'fetch_cost': 0          # c_f = 3
                }

            sample_user_requests = []
            for uid in range(1, nums_user):  # Generate a reasonable number of potential users
                user_lat = random.uniform(-90, 90)
                user_lon = random.uniform(-180, 180)
                requested_view = random.randint(1, nums_views)
                sample_user_requests.append((uid, user_lat, user_lon, requested_view))

            # Get user requests and create request range R_n(t) = [h, l]
            user_requests = sat.receive_requests(current_time, sample_user_requests, request_range_size=5)

            # Process user requests in the area
            # user_requests = sat.receive_requests(current_time, [], request_range_size=5)

            # For each requested view, check if it's in our cache
            for view in new_views:
                # Determine if this view was explicitly requested by a user
                is_requested = False
                for req in user_requests.values():
                    if view == req['requested_view'] or view in req['view_range']:
                        is_requested = True
                        break
                
                # Always apply costs, not just for requested views
                # If we have it in cache, just transmission cost
                if view in current_cache:
                    sat.cost_stats['transmission_cost'] += 2  # c_s = 2
                else:
                    # Cache miss: apply all three costs
                    sat.cost_stats['cache_miss_cost'] += 10  # c_m = 10
                    sat.cost_stats['fetch_cost'] += 3       # c_f = 3
                    sat.cost_stats['transmission_cost'] += 2  # c_s = 2
            
            # For each new view being added
            for view in new_views:
                # If cache is full, remove least recently used item
                if len(current_cache) >= sat.max_cache_size:
                    # Remove the least recently used item (first item in LRU history)
                    if sat.lru_history:
                        least_recent = sat.lru_history.pop(0)
                        current_cache.remove(least_recent)
                    elif current_cache:
                        # If LRU history is empty but cache has items, remove a random one
                        least_recent = random.choice(list(current_cache))
                        current_cache.remove(least_recent)
                
                # Add the new view to cache
                current_cache.add(view)
                
                # Add to end of LRU history (most recently used)
                sat.lru_history.append(view)
        
        # Make sure LRU history only contains currently cached items
        sat.lru_history = [v for v in sat.lru_history if v in current_cache]
        
        # Ensure all cached items are in the LRU history
        for v in current_cache:
            if v not in sat.lru_history:
                sat.lru_history.append(v)
        
        # Update the cache state
        sat.update_cache(next_time, current_cache)
    
    return next_time

def LFU(satellites, current_time, update_position_func):
    """
    Function to evolve satellite system to the next time step using LFU cache replacement.
    
    Parameters:
    - satellites: List of SatelliteWithCache objects
    - current_time: Current time step
    - update_position_func: Function to update satellite positions
    
    Returns:
    - Next time step
    """
    next_time = current_time + 1
    
    # Update positions based on orbital mechanics
    new_positions = update_position_func(next_time)
    
    for sat, new_pos in zip(satellites, new_positions):
        # Update position
        sat.update_position(next_time, new_pos)
        
        # Get current cache state
        current_cache = sat.get_cache(current_time)
        
        # Initialize frequency counter if it doesn't exist
        if not hasattr(sat, 'frequency_counter'):
            sat.frequency_counter = {view: 1 for view in current_cache}
        
        # Determine remaining views that could be added to cache
        remaining_views = universal_views - current_cache
        
        # Decide how many new views to fetch (random number between 0 and 3)
        num_new_views = random.randint(0, min(3, len(remaining_views)))
        
        if num_new_views > 0 and remaining_views:
            # Select random new views to add
            new_views = set(random.sample(list(remaining_views), num_new_views))
            
            # Initialize cost counters for this satellite if they don't exist
            if not hasattr(sat, 'cost_stats'):
                sat.cost_stats = {
                    'transmission_cost': 0,  # c_s = 2
                    'cache_miss_cost': 0,    # c_m = 10
                    'fetch_cost': 0          # c_f = 3
                }

            sample_user_requests = []
            for uid in range(1, nums_user):  # Generate a reasonable number of potential users
                user_lat = random.uniform(-90, 90)
                user_lon = random.uniform(-180, 180)
                requested_view = random.randint(1, nums_views)
                sample_user_requests.append((uid, user_lat, user_lon, requested_view))

            # Get user requests and create request range R_n(t) = [h, l]
            user_requests = sat.receive_requests(current_time, sample_user_requests, request_range_size=5)
            # Process user requests in the area
            # user_requests = sat.receive_requests(current_time, [], request_range_size=3)

            # For each requested view, check if it's in our cache
            for view in new_views:
                # Determine if this view was explicitly requested by a user
                is_requested = False
                for req in user_requests.values():
                    if view == req['requested_view'] or view in req['view_range']:
                        is_requested = True
                        break
                
                # Always apply costs, not just for requested views
                # If we have it in cache, just transmission cost
                if view in current_cache:
                    sat.cost_stats['transmission_cost'] += 2  # c_s = 2
                    # Increment frequency counter for this view
                    sat.frequency_counter[view] = sat.frequency_counter.get(view, 0) + 1
                else:
                    # Cache miss: apply all three costs
                    sat.cost_stats['cache_miss_cost'] += 10  # c_m = 10
                    sat.cost_stats['fetch_cost'] += 3        # c_f = 3
                    sat.cost_stats['transmission_cost'] += 2 # c_s = 2
            
            # For each new view being added
            for view in new_views:
                # If cache is full, remove least frequently used item
                if len(current_cache) >= sat.max_cache_size:
                    # Check if frequency counter has items before finding min
                    if sat.frequency_counter:
                        # Find least frequently used item
                        lfu_item = min(sat.frequency_counter.items(), key=lambda x: x[1])[0]
                        
                        # If there are multiple items with the same frequency, choose randomly
                        min_freq = sat.frequency_counter[lfu_item]
                        lfu_candidates = [v for v, f in sat.frequency_counter.items() if f == min_freq]
                        if len(lfu_candidates) > 1:
                            lfu_item = random.choice(lfu_candidates)
                        
                        # Remove the LFU item
                        current_cache.remove(lfu_item)
                        del sat.frequency_counter[lfu_item]
                    elif current_cache:
                        # If no frequency info but cache has items, remove a random one
                        lfu_item = random.choice(list(current_cache))
                        current_cache.remove(lfu_item)
                
                # Add the new view to cache
                current_cache.add(view)
                
                # Initialize frequency counter for new view
                sat.frequency_counter[view] = 1
        
        # Update the cache state
        sat.update_cache(next_time, current_cache)
    
    return next_time

def online_algorithm(satellites, current_time, update_position_func):
    """
    Implementation of LEO Satellite Cooperative Caching algorithm as shown in the image.
    
    Parameters:
    - satellites: List of SatelliteWithCache objects
    - current_time: Current time step
    - update_position_func: Function to update satellite positions
    
    Returns:
    - Next time step
    """
    next_time = current_time + 1
    
    # Update positions based on orbital mechanics
    new_positions = update_position_func(next_time)
    
    for n, (sat, new_pos) in enumerate(zip(satellites, new_positions)):
        # Update position
        sat.update_position(next_time, new_pos)
        
        # Get current cache state V_n(t)
        V_n = sat.get_cache(current_time)
        
        # Get neighbor caches {V_{n-1}(t), V_{n+1}(t)}
        left_neighbor_idx = (n - 1) % len(satellites)
        right_neighbor_idx = (n + 1) % len(satellites)
        V_n_minus_1 = satellites[left_neighbor_idx].get_cache(current_time)
        V_n_plus_1 = satellites[right_neighbor_idx].get_cache(current_time)
        
        # Generate random user requests for this time step and area
        sample_user_requests = []
        for uid in range(1, nums_user):  # Generate a reasonable number of potential users
            user_lat = random.uniform(-90, 90)
            user_lon = random.uniform(-180, 180)
            requested_view = random.randint(1, nums_views)
            sample_user_requests.append((uid, user_lat, user_lon, requested_view))

        # Get user requests and create request range R_n(t) = [h, l]
        user_requests = sat.receive_requests(current_time, sample_user_requests, request_range_size=5)
        
        # Extract requested views to determine range [h, l]
        requested_views = set()
        for req in user_requests.values():
            requested_views.add(req['requested_view'])
            requested_views.update(req['view_range'])
        
        if requested_views:
            h = min(requested_views)
            l = max(requested_views)
            R_n = list(range(h, l + 1))
        else:
            # Default range if no requests
            h, l = 1, min(10, nums_views)
            R_n = list(range(h, l + 1))
        
        # Initialize cost statistics if not present
        if not hasattr(sat, 'cost_stats'):
            sat.cost_stats = {
                'transmission_cost': 0,  # c_n,s = 2
                'cache_miss_cost': 0,    # c_m = 10
                'fetch_cost': 0,         # c_n,f = 3
                'dibr_cost': 0,          # c_dibr = 5
                'isl_cost': 0            # c_isl = 1
            }
        
        # Initialize view sets if they don't exist
        if not hasattr(sat, 'views_f'):
            sat.views_f = set()      # V^f: views fetched from ground
        if not hasattr(sat, 'views_DIBR'):
            sat.views_DIBR = set()   # V^DIBR: views from DIBR
        if not hasattr(sat, 'views_ISL'):
            sat.views_ISL = set()    # V^ISL: views from ISL
        if not hasattr(sat, 'views_s'):
            sat.views_s = set()      # V^s: shared views
        if not hasattr(sat, 'views_e'):
            sat.views_e = set()      # V^e: existing views
            
        # Calculate the possibility of view v as P(v)
        # For simplicity, we'll use frequency-based popularity
        view_popularity = {v: 0 for v in range(h, l + 1)}
        for req in user_requests.values():
            view_popularity[req['requested_view']] = view_popularity.get(req['requested_view'], 0) + 1
            
        # Normalize to get P(v)
        total_requests = sum(view_popularity.values()) or 1  # Avoid division by zero
        P_v = {v: count/total_requests for v, count in view_popularity.items()}
        
        # Initialize view costs and utility
        tau_j = {}
        mu_h_j = {}
        
        # For all j in range [h, l]
        for j in R_n:
            # Calculate tau_j based on where the view is available
            if j in V_n:
                # If j is in local cache
                tau_j[j] = 2  # c_n,s = 2
            elif j in (V_n_minus_1.union(V_n_plus_1) - V_n):
                # If j is in neighbor caches but not in local cache
                tau_j[j] = 1 + 2  # c_ISL + c_n,s
                sat.views_ISL.add(j)
            else:
                # If j is not available in local or neighbor caches
                tau_j[j] = 10 + 3 + 2  # c_m + c_n,f + c_n,s
                sat.views_f.add(j)
            
            # Calculate utility μ_h,j (cost-weighted by popularity)
            mu_h_j[j] = P_v.get(j, 0) * tau_j[j]
        
        # Find optimal sets that minimize the utility function
        # Sort views by utility (lower is better)
        # Calculate the optimal μ_h,j values using dynamic programming
        alpha = 1.2  # Parameter for view distance cost
        T_DIBR = 1   # DIBR cost constant from satellite's cost_values
        D = 3        # Maximum distance for DIBR (adjust based on requirements)

        # Initialize dynamic programming table
        dp = {}  # μ_h,j values

        # Base case: μ_h,h = τ_h
        dp[h] = tau_j[h]

        # Fill the dp table for all j > h
        for j in range(h + 1, l + 1):
            # Try all possible previous views i
            min_cost = float('inf')
            
            for i in range(max(j - D, h), j):
                # Calculate cost if we use view i and DIBR for views between i and j
                dibr_cost = (alpha * (j - i) + T_DIBR) * (j - i - 1)
                cost = tau_j[j] + dp[i] + dibr_cost
                
                # Update minimum cost
                min_cost = min(min_cost, cost)
            
            # If no valid i exists, use direct cost
            if min_cost == float('inf'):
                min_cost = tau_j[j]
            
            dp[j] = min_cost

        # No need to apply P_v weighting here - that's done separately
        mu_h_j = dp
        sorted_views = sorted(mu_h_j.keys(), key=lambda x: mu_h_j[x])
        
        # Initialize optimal sets
        V_f_optimal = set()
        V_DIBR_optimal = set()
        V_ISL_optimal = set()
        V_s_optimal = set()
        
        # Populate optimal sets based on availability and utility
        new_cache = set()
        for view in sorted_views:
            if len(new_cache) >= sat.max_cache_size:
                break
                
            # Determine how to get this view
            if view in V_n:
                # Already in cache
                new_cache.add(view)
            elif view in (V_n_minus_1.union(V_n_plus_1) - V_n):
                # Available via ISL
                V_ISL_optimal.add(view)
                new_cache.add(view)
                sat.cost_stats['isl_cost'] += 1
                sat.cost_stats['transmission_cost'] += 2
            elif any(abs(view - v) <= 2 for v in V_n):
                # Can use DIBR
                V_DIBR_optimal.add(view)
                new_cache.add(view)
                sat.cost_stats['dibr_cost'] += 2
                sat.cost_stats['transmission_cost'] += 2
            else:
                # Need to fetch from ground
                V_f_optimal.add(view)
                new_cache.add(view)
                sat.cost_stats['cache_miss_cost'] += 10
                sat.cost_stats['fetch_cost'] += 3
                sat.cost_stats['transmission_cost'] += 2
        
        # If cache space is not sufficient, use P(v) to decide which to keep
        if len(new_cache) > sat.max_cache_size:
            # Sort by popularity and keep only the top max_cache_size items
            V_e_optimal = set(sorted(new_cache, key=lambda v: P_v.get(v, 0), reverse=True)[:sat.max_cache_size])
        else:
            V_e_optimal = new_cache
        
        # Update the satellite's view sets
        sat.views_f = V_f_optimal
        sat.views_DIBR = V_DIBR_optimal
        sat.views_ISL = V_ISL_optimal
        sat.views_e = V_e_optimal
        
        # Update the cache state for next time step
        sat.update_cache(next_time, V_e_optimal)
        
        # Update the satellite view sets for this time step
        sat.update_view_sets(next_time, {
            'f': V_f_optimal,
            'DIBR': V_DIBR_optimal,
            'ISL': V_ISL_optimal,
            'e': V_e_optimal,
            's': V_s_optimal
        })
    
    return next_time

# Initialize satellites in a ring formation
def initialize_ring_constellation(num_satellites, inclination=45, altitude=500):
    """
    1. Position satellites in a ring formation around Earth.
    2. Parameters:
        - num_satellites: Number of satellites in the constellation
        - inclination: Inclination of the orbital plane in degrees
        - altitude: Altitude of the satellites in km
    3. Returns:
        - List of (longitude, latitude) tuples for each satellite
        - Function to update positions for a given time step
    """
    # Convert inclination to radians
    incl_rad = math.radians(inclination)
    
    # Earth radius in km
    earth_radius = 6371
    
    # Orbital radius (Earth radius + altitude)
    orbital_radius = earth_radius + altitude
    
    # Calculate positions evenly spaced around the orbit
    positions = []
    for i in range(num_satellites):
        # Angular position in the orbit (0 to 2π)
        angle = 2 * math.pi * i / num_satellites
        
        # Calculate position in orbital plane
        x = orbital_radius * math.cos(angle)
        y = orbital_radius * math.sin(angle)
        
        # Rotate by inclination around the x-axis
        y_rotated = y * math.cos(incl_rad)
        z_rotated = y * math.sin(incl_rad)
        
        # Convert to longitude and latitude (in degrees)
        lon = math.degrees(math.atan2(y_rotated, x))
        lat = math.degrees(math.asin(z_rotated / orbital_radius))
        
        positions.append((lon, lat))
    
    def update_positions(time_step, angular_velocity=2*math.pi/timeSlot):
        """
        Update satellite positions for the given time step.
        
        Parameters:
        - time_step: Current time step (0 to timeSlot-1)
        - angular_velocity: Angular velocity in radians per time step
        
        Returns:
        - Updated list of (longitude, latitude) positions
        """
        # Calculate phase offset for this time step
        phase_offset = time_step * angular_velocity
        
        updated_positions = []
        for i in range(num_satellites):
            # Angular position with time-based offset
            angle = (2 * math.pi * i / num_satellites) + phase_offset
            
            # Calculate position in orbital plane
            x = orbital_radius * math.cos(angle)
            y = orbital_radius * math.sin(angle)
            
            # Rotate by inclination around the x-axis
            y_rotated = y * math.cos(incl_rad)
            z_rotated = y * math.sin(incl_rad)
            
            # Convert to longitude and latitude
            lon = math.degrees(math.atan2(y_rotated, x))
            lat = math.degrees(math.asin(z_rotated / orbital_radius))
            
            updated_positions.append((lon, lat))
        
        return updated_positions
    # Return initial positions and the update function
    return positions, update_positions

class SatelliteWithCache: 
    def __init__(self, sat_id, initial_position, max_cache_size):
        self.sat_id = sat_id
        self.positions = {0: initial_position}  # Track positions over time
        self.cache = {0: set()}                 # Cache state at each time slot
        self.max_cache_size = max_cache_size
        # Additional satellite characteristics
        self.orbit_height = 500  # km, default orbit height
        self.inclination = 45    # degrees
        self.coverage_angle = None  # Will be calculated
        self.service_area = set()  # Points covered by this satellite
        self.power_level = 100     # Percentage of full power
        self.transmit_power = 20   # W (watts)
        self.bandwidth = 1000      # Mbps
        self.latency = 10          # ms
        self.fuel = 100            # Percentage remaining
        self.operational = True    # Operational status

        # Initialize tracking and cost stats properly
        self.lru_history = []      # Track LRU history for cache replacement
        self.frequency_counter = {}  # Track frequency for LFU cache replacement
        
        # Initialize cost tracking properly - these are cumulative counters not constants
        self.cost_stats = {
            'transmission_cost': 0,  # c_s = 2
            'cache_miss_cost': 0,    # c_m = 10
            'fetch_cost': 0,         # c_f = 3
            'dibr_cost': 0,          # c_dibr = 5
            'isl_cost': 0            # c_isl = 1
        }
        
        # Cost constants
        self.cost_values = {
            'transmission': 3,  # c_s = 2
            'cache_miss': 10,   # c_m = 10
            'fetch': 5,         # c_f = 3
            'dibr': 1,          # c_dibr = 3
            'isl': 3            # c_isl = 1
        }
        
        self.view_stats = {}       # For tracking view frequency and recency

        ########## View sets for the satellite ##########
        self.views_syn = set()    # V^{syn}_n(t): Synthetic views generated by the satellite
        self.views_f = set()      # V^f_n(t): Views fetched from ground stations
        self.views_DIBR = set()   # V^{DIBR}_n(t): Views generated through depth-image-based rendering
        self.views_e = set()      # V^e_n(t): Existing views in satellite cache
        self.views_s = set()      # V^s_n(t): Views shared from neighboring satellites
        self.views_ISL = set()    # V^{ISL}_n(t): Views available via inter-satellite links
        
        # Per-timestep cost tracking for better analysis
        self.time_costs = {}
                
    def update_view_sets(self, time_slot, view_sets=None):
        """
        Update the satellite's view sets for a given time slot.
        
        Parameters:
        - time_slot: Current time slot
        - view_sets: Dictionary containing updated view sets
        """
        if view_sets is None:
            return
            
        # Update individual view sets if provided
        if 'syn' in view_sets:
            self.views_syn = set(view_sets['syn'])
        if 'f' in view_sets:
            self.views_f = set(view_sets['f'])
        if 'DIBR' in view_sets:
            self.views_DIBR = set(view_sets['DIBR'])
        if 'e' in view_sets:
            self.views_e = set(view_sets['e'])
        if 's' in view_sets:
            self.views_s = set(view_sets['s'])
        if 'ISL' in view_sets:
            self.views_ISL = set(view_sets['ISL'])
            
        # Update the total cache based on all available views
        combined_views = self.views_syn | self.views_f | self.views_DIBR | self.views_e | self.views_s | self.views_ISL
        
        # Ensure cache doesn't exceed max size
        if len(combined_views) > self.max_cache_size:
            combined_views = set(list(combined_views)[:self.max_cache_size])
            
        # Update the overall cache for this time slot
        self.update_cache(time_slot, combined_views)

    def add_cost(self, time_slot, cost_type, amount=1):
        """
        Add cost of specified type to the satellite's cost statistics.
        
        Parameters:
        - time_slot: Current time slot
        - cost_type: Type of cost ('transmission', 'cache_miss', 'fetch', 'dibr', 'isl')
        - amount: Number of times this cost is incurred (default 1)
        """
        # Map cost type to the corresponding key in cost_stats
        cost_key = f'{cost_type}_cost'
        
        # Get the cost value for this type
        cost_value = self.cost_values.get(cost_type, 0) * amount
        
        # Add to the cumulative cost stats
        if cost_key in self.cost_stats:
            self.cost_stats[cost_key] += cost_value
        
        # Track costs per time slot
        if time_slot not in self.time_costs:
            self.time_costs[time_slot] = {}
        
        if cost_key not in self.time_costs[time_slot]:
            self.time_costs[time_slot][cost_key] = 0
            
        self.time_costs[time_slot][cost_key] += cost_value
        
        return cost_value

    def receive_requests(self, time_slot, user_locations, request_range_size=5):
        """
        Process view requests from users within the satellite's coverage area.
        
        Parameters:
        - time_slot: Current time slot
        - user_locations: List of (user_id, lat, lon, requested_view) tuples
        - request_range_size: Size of the request range (B)
        
        Returns:
        - Dictionary of received requests with user IDs as keys and request details as values
        """
        # Get satellite position at this time slot
        current_position = self.get_position(time_slot)
        
        # Calculate coverage area
        coverage_data = area_model([current_position], psi=self.coverage_angle or math.radians(25))
        
        # Initialize empty dictionary for received requests
        received_requests = {}
        
        # Check each user request
        for user_id, lat, lon, requested_view in user_locations:
            # Check if user is in coverage area
            if is_point_covered(lat, lon, coverage_data):
                # Calculate the request range
                half_range = request_range_size // 2
                min_view = max(1, requested_view - half_range)
                max_view = min(nums_views, requested_view + half_range)
                view_range = list(range(min_view, max_view + 1))
                
                # Store the request details
                received_requests[user_id] = {
                    'user_location': (lat, lon),
                    'requested_view': requested_view,
                    'view_range': view_range,
                    'time_slot': time_slot,
                    'can_serve': any(view in self.get_cache(time_slot) for view in view_range)
                }
                
        return received_requests
    
    def can_serve_request(self, time_slot, requested_view, request_range_size=5):
        """
        Check if the satellite can serve a requested view with its current cache.
        
        Parameters:
        - time_slot: The current time slot
        - requested_view: The main requested view ID
        - request_range_size: Size of the request range (B)
        
        Returns:
        - True if any view in the request range is in the cache, False otherwise
        """
        half_range = request_range_size // 2
        min_view = max(1, requested_view - half_range)
        max_view = min(nums_views, requested_view + half_range)
        
        # Get current cache
        current_cache = self.get_cache(time_slot)
        
        # Check if any view in the range is in the cache
        for view_id in range(min_view, max_view + 1):
            if view_id in current_cache:
                return True
        
        return False
    
    def update_position(self, time_slot, new_position):
        """Update the satellite position for a given time slot"""
        self.positions[time_slot] = new_position
    
    def update_cache(self, time_slot, new_cache_contents):
        """Update the satellite cache for a given time slot"""
        # Ensure cache size doesn't exceed maximum
        if len(new_cache_contents) > self.max_cache_size:
            new_cache_contents = set(list(new_cache_contents)[:self.max_cache_size])
        self.cache[time_slot] = new_cache_contents
    
    def get_position(self, time_slot):
        """Get satellite position at a specific time slot"""
        return self.positions.get(time_slot, self.positions[max(self.positions.keys())])
    
    def get_cache(self, time_slot):
        """Get cache contents at a specific time slot"""
        return self.cache.get(time_slot, self.cache[max(self.cache.keys())])
    
    def get_costs_at_time(self, time_slot):
        """Get costs incurred at a specific time slot"""
        return self.time_costs.get(time_slot, {})
        
    def update_parameters(self, time_slot, params=None):
        """Update satellite parameters for a given time slot"""
        if params is None:
            # If no specific parameters are provided, make random adjustments
            self.power_level = max(0, min(100, self.power_level + random.uniform(-2, 1)))
            self.transmit_power = max(5, min(30, self.transmit_power + random.uniform(-0.5, 0.5)))
            self.bandwidth = max(500, min(2000, self.bandwidth + random.uniform(-50, 50)))
            self.latency = max(5, min(20, self.latency + random.uniform(-0.5, 0.5)))
            self.fuel = max(0, self.fuel - random.uniform(0.01, 0.05))
            
            # Update operational status based on fuel and power
            self.operational = self.fuel > 0 and self.power_level > 10
        else:
            # Update with provided parameters
            if 'power_level' in params:
                self.power_level = params['power_level']
            if 'transmit_power' in params:
                self.transmit_power = params['transmit_power']
            if 'bandwidth' in params:
                self.bandwidth = params['bandwidth']
            if 'latency' in params:
                self.latency = params['latency']
            if 'fuel' in params:
                self.fuel = params['fuel']
            if 'operational' in params:
                self.operational = params['operational']
                
        # Calculate new coverage based on current parameters
        effective_power = self.transmit_power * (self.power_level / 100)
        coverage_adjustment = math.sqrt(effective_power / 20)  # Scale coverage by power square root
        
        # Update coverage angle based on power (more power = wider coverage)
        base_angle = math.radians(25)  # Base coverage angle
        self.coverage_angle = base_angle * coverage_adjustment
    
    def __str__(self):
        latest_time = max(self.positions.keys())
        latest_pos = self.positions[latest_time]
        cache_size = len(self.cache[latest_time])
        return f"Satellite {self.sat_id} at ({latest_pos[0]:.2f}, {latest_pos[1]:.2f}) with {cache_size} cached views"

# Simulation parameters
nums_user = 1000    #(K = 1000)
nums_sat = 20       #(N = 10)
nums_ground = 40    #(G = 10)
timeSlot = 1000   #(T = 1000)
max_cache_size = 60   # Every LEO satellite have the same cache storage (Z = 5)

# Define the universal set of views V = {1, 2, ..., v}
nums_views = 40000  # Total number of views
universal_views = set(range(1, nums_views + 1))  # V = {1, 2, ..., v}

# Initialize satellites with caches
satellites = []
initial_positions, update_position_func = initialize_ring_constellation(nums_sat)

# Create satellite objects
for sat_id in range(1, nums_sat + 1):
    sat = SatelliteWithCache(
        sat_id=sat_id,
        initial_position=initial_positions[sat_id-1],
        max_cache_size=max_cache_size
    )
    satellites.append(sat)

print("Initial satellite states:")
for sat in satellites:
    print(sat)

print(sat.positions)


# Calculate coverage parameters
height = 500  # km
theta_min = math.radians(10)  # 10 degrees minimum elevation angle
coverage_params = coverage_model(height, theta_min)
print(f"Coverage angle (psi): {math.degrees(coverage_params['psi']):.2f} degrees")
print(f"Theoretical coverage area per satellite: {coverage_params['coverage_area']:.2f} km²")

# Initialize satellite service areas
current_time = 0
satellite_positions = [sat.get_position(current_time) for sat in satellites]
satellite_coverage = area_model(satellite_positions, psi=coverage_params['psi'])

# # Initialize random caches for time step 0
# for sat in satellites:
#     # Generate random initial cache (between 1 and max_cache_size items)
#     cache_size = random.randint(1, sat.max_cache_size)
#     initial_cache = set(random.sample(list(universal_views), cache_size))
#     sat.update_cache(0, initial_cache)

# # Advance system to next time slot
# next_time = LRU(satellites, current_time, update_position_func)
# print(sat.positions)

# # Print satellite cache states at the next time slot
# print("\n===== Satellite cache states at time slot", next_time, " =====")
# # Generate random user requests for the first time slot
# print("\nGenerating random user requests for time slot", next_time)
# user_requests = []
# for user_id in range(1, nums_user + 1):
#     # Random user location
#     user_lat = random.uniform(-90, 90)
#     user_lon = random.uniform(-180, 180)
#     # Random requested view
#     requested_view = random.randint(1, nums_views)
#     user_requests.append((user_id, user_lat, user_lon, requested_view))

# # Process user requests at each satellite
# satellite_requests = {}
# for sat in satellites:
#     # Get received requests from users within coverage area
#     received_reqs = sat.receive_requests(next_time, user_requests)
#     satellite_requests[sat.sat_id] = received_reqs
    
#     # Print satellite status and cache information
#     cache = sat.get_cache(next_time)
#     position = sat.get_position(next_time)
#     print(f"Satellite {sat.sat_id} at ({position[0]:.2f}, {position[1]:.2f}): Cache contains {len(cache)} items")
#     print(f"    -> Cached views: {sorted(list(cache))}")
#     print(f"    -> Received {len(received_reqs)} user requests")
    
#     # Print sample of requests (max 5)
#     sample_reqs = list(received_reqs.items())[:5]
#     for user_id, req_data in sample_reqs:
#         can_serve = req_data['can_serve']
#         status = "CAN SERVE" if can_serve else "CANNOT SERVE"
#         print(f"        User {user_id} requested view {req_data['requested_view']} - {status}")




# Run simulations and compare LRU, LFU, and online algorithm
import matplotlib.pyplot as plt

# Reset satellites for new simulations
lru_satellites = []
lfu_satellites = []
online_satellites = []

# Create new satellite objects for each algorithm
for sat_id in range(1, nums_sat + 1):
    # Generate identical initial cache for fair comparison
    cache_size = random.randint(1, max_cache_size)
    initial_cache = set(random.sample(list(universal_views), cache_size))
    
    # For LRU
    lru_sat = SatelliteWithCache(
        sat_id=sat_id,
        initial_position=initial_positions[sat_id-1],
        max_cache_size=max_cache_size
    )
    lru_sat.update_cache(0, initial_cache.copy())
    lru_satellites.append(lru_sat)
    
    # For LFU
    lfu_sat = SatelliteWithCache(
        sat_id=sat_id,
        initial_position=initial_positions[sat_id-1],
        max_cache_size=max_cache_size
    )
    lfu_sat.update_cache(0, initial_cache.copy())
    lfu_satellites.append(lfu_sat)
    
    # For online algorithm
    online_sat = SatelliteWithCache(
        sat_id=sat_id,
        initial_position=initial_positions[sat_id-1],
        max_cache_size=max_cache_size
    )
    online_sat.update_cache(0, initial_cache.copy())
    online_satellites.append(online_sat)

# Run simulations
num_time_slots = 500  # Run for 500 time slots
lru_time = 0
lfu_time = 0
online_time = 0

lru_costs = [0]
lfu_costs = [0]
online_costs = [0]
time_slots = list(range(num_time_slots + 1))

# Initialize cost tracking
lru_accumulated_costs = {'transmission_cost': 0, 'cache_miss_cost': 0, 'fetch_cost': 0}
lfu_accumulated_costs = {'transmission_cost': 0, 'cache_miss_cost': 0, 'fetch_cost': 0}
online_accumulated_costs = {
    'transmission_cost': 0, 
    'cache_miss_cost': 0, 
    'fetch_cost': 0, 
    'dibr_cost': 0, 
    'isl_cost': 0
}

print("\n===== Running simulations to compare LRU, LFU, and online algorithm =====")

# Simulate all three algorithms
for i in range(num_time_slots):
    # Advance LRU satellites
    lru_time = LRU(lru_satellites, lru_time, update_position_func)
    
    # Calculate LRU cost for this time slot
    lru_slot_cost = 0
    for sat in lru_satellites:
        if hasattr(sat, 'cost_stats'):
            current_transmission = sat.cost_stats['transmission_cost'] - lru_accumulated_costs['transmission_cost']
            current_cache_miss = sat.cost_stats['cache_miss_cost'] - lru_accumulated_costs['cache_miss_cost'] 
            current_fetch = sat.cost_stats['fetch_cost'] - lru_accumulated_costs['fetch_cost']
            
            # Update accumulated costs
            lru_accumulated_costs['transmission_cost'] += current_transmission
            lru_accumulated_costs['cache_miss_cost'] += current_cache_miss
            lru_accumulated_costs['fetch_cost'] += current_fetch
            
            # Add to slot cost
            lru_slot_cost += (current_transmission + current_cache_miss + current_fetch)
    
    # Accumulate costs
    lru_costs.append(lru_costs[-1] + lru_slot_cost)
    
    # Advance LFU satellites
    lfu_time = LFU(lfu_satellites, lfu_time, update_position_func)
    
    # Calculate LFU cost for this time slot
    lfu_slot_cost = 0
    for sat in lfu_satellites:
        if hasattr(sat, 'cost_stats'):
            current_transmission = sat.cost_stats['transmission_cost'] - lfu_accumulated_costs['transmission_cost']
            current_cache_miss = sat.cost_stats['cache_miss_cost'] - lfu_accumulated_costs['cache_miss_cost'] 
            current_fetch = sat.cost_stats['fetch_cost'] - lfu_accumulated_costs['fetch_cost']
            
            # Update accumulated costs
            lfu_accumulated_costs['transmission_cost'] += current_transmission
            lfu_accumulated_costs['cache_miss_cost'] += current_cache_miss
            lfu_accumulated_costs['fetch_cost'] += current_fetch
            
            # Add to slot cost
            lfu_slot_cost += (current_transmission + current_cache_miss + current_fetch)
    
    # Accumulate costs
    lfu_costs.append(lfu_costs[-1] + lfu_slot_cost)
    
    # Advance online algorithm satellites
    online_time = online_algorithm(online_satellites, online_time, update_position_func)
    
    # Calculate online algorithm cost for this time slot
    online_slot_cost = 0
    for sat_idx, sat in enumerate(online_satellites):
        # Make sure cost_stats exists
        if not hasattr(sat, 'cost_stats'):
            sat.cost_stats = {
                'transmission_cost': 0, 
                'cache_miss_cost': 0, 
                'fetch_cost': 0,
                'dibr_cost': 0,
                'isl_cost': 0
            }
            
        # Initialize attributes needed by the online algorithm if they don't exist
        if not hasattr(sat, 'views_ISL'):
            sat.views_ISL = set()
        if not hasattr(sat, 'views_DIBR'):
            sat.views_DIBR = set()
        if not hasattr(sat, 'views_f'):
            sat.views_f = set()
        
        # Process user requests to generate some costs
        user_requests = sat.receive_requests(online_time, [], request_range_size=3)
        current_cache = sat.get_cache(online_time)
        
        # Add some costs for processing - simulate cost addition that should happen in online_algorithm
        for req_id, req in user_requests.items():
            if req['requested_view'] in current_cache:
                # Transmission cost for serving from cache
                sat.cost_stats['transmission_cost'] += 2
            else:
                # Cache miss costs
                sat.cost_stats['cache_miss_cost'] += 10
                
                # Randomly decide how to fetch the missing view
                fetch_method = random.choice(['dibr', 'isl', 'fetch'])
                if fetch_method == 'dibr':
                    sat.cost_stats['dibr_cost'] += 5
                    sat.views_DIBR.add(req['requested_view'])
                elif fetch_method == 'isl':
                    sat.cost_stats['isl_cost'] += 1
                    sat.views_ISL.add(req['requested_view'])
                else:
                    sat.cost_stats['fetch_cost'] += 3
                    sat.views_f.add(req['requested_view'])
                
                # Always add transmission cost
                sat.cost_stats['transmission_cost'] += 2
        
        # Calculate the deltas from last accumulated costs
        current_transmission = sat.cost_stats['transmission_cost'] - online_accumulated_costs['transmission_cost']
        current_cache_miss = sat.cost_stats['cache_miss_cost'] - online_accumulated_costs['cache_miss_cost'] 
        current_fetch = sat.cost_stats['fetch_cost'] - online_accumulated_costs['fetch_cost']
        current_dibr = sat.cost_stats.get('dibr_cost', 0) - online_accumulated_costs['dibr_cost']
        current_isl = sat.cost_stats.get('isl_cost', 0) - online_accumulated_costs['isl_cost']
        
        # Update accumulated costs
        online_accumulated_costs['transmission_cost'] += current_transmission
        online_accumulated_costs['cache_miss_cost'] += current_cache_miss
        online_accumulated_costs['fetch_cost'] += current_fetch
        online_accumulated_costs['dibr_cost'] += current_dibr
        online_accumulated_costs['isl_cost'] += current_isl
        
        # Add to slot cost
        slot_cost = current_transmission + current_cache_miss + current_fetch + current_dibr + current_isl
        online_slot_cost += slot_cost
        
        # Debug print for the first satellite periodically
        # if sat_idx == 0 and i % 50 == 0:
        #     print(f"Online sat 0 at time {i}: slot_cost={slot_cost}, trans={current_transmission}, "
        #           f"miss={current_cache_miss}, fetch={current_fetch}, dibr={current_dibr}, isl={current_isl}")
        #     online_slot_cost += (current_transmission + current_cache_miss + 
        #                        current_fetch + current_dibr + current_isl)
    
    # Accumulate costs
    online_costs.append(online_costs[-1] + online_slot_cost)
    
    # Print status every 50 time slots
    if (i + 1) % 50 == 0:
        print(f"Completed {i + 1} time slots:")
        print(f"  LRU cost: {lru_costs[-1]}")
        print(f"  LFU cost: {lfu_costs[-1]}")
        print(f"  Online algorithm cost: {online_costs[-1]}")

# Plot 1: Accumulated costs over time for all three algorithms
plt.figure(figsize=(12, 7))
plt.plot(time_slots, lru_costs, 'b-', linewidth=2, label='LRU')
plt.plot(time_slots, lfu_costs, 'r-', linewidth=2, label='LFU')
plt.plot(time_slots, online_costs, 'g-', linewidth=2, label='Online Algorithm')
plt.title('Comparison of Accumulated Costs')
plt.xlabel('Time Slot')
plt.ylabel('Accumulated Cost')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('algorithm_comparison_accumulated_costs.png')
plt.show()

# Plot 2: Calculate total costs and cost components for each algorithm
lru_transmission = lru_accumulated_costs['transmission_cost']
lru_cache_miss = lru_accumulated_costs['cache_miss_cost']
lru_fetch = lru_accumulated_costs['fetch_cost']
lru_total = lru_transmission + lru_cache_miss + lru_fetch

lfu_transmission = lfu_accumulated_costs['transmission_cost']
lfu_cache_miss = lfu_accumulated_costs['cache_miss_cost']
lfu_fetch = lfu_accumulated_costs['fetch_cost']
lfu_total = lfu_transmission + lfu_cache_miss + lfu_fetch

online_transmission = online_accumulated_costs['transmission_cost']
online_cache_miss = online_accumulated_costs['cache_miss_cost']
online_fetch = online_accumulated_costs['fetch_cost']
online_dibr = online_accumulated_costs['dibr_cost']
online_isl = online_accumulated_costs['isl_cost']
online_total = online_transmission + online_cache_miss + online_fetch + online_dibr + online_isl

# Create a bar chart comparing component costs
labels = ['Transmission', 'Cache Miss', 'Fetch', 'DIBR', 'ISL', 'Total']
lru_values = [lru_transmission, lru_cache_miss, lru_fetch, 0, 0, lru_total]
lfu_values = [lfu_transmission, lfu_cache_miss, lfu_fetch, 0, 0, lfu_total]
online_values = [online_transmission, online_cache_miss, online_fetch, online_dibr, online_isl, online_total]

x = np.arange(len(labels))
width = 0.25

fig, ax = plt.subplots(figsize=(14, 8))
rects1 = ax.bar(x - width, lru_values, width, label='LRU', color='blue')
rects2 = ax.bar(x, lfu_values, width, label='LFU', color='red')
rects3 = ax.bar(x + width, online_values, width, label='Online Algorithm', color='green')

ax.set_ylabel('Cost')
ax.set_title('Cost Components Comparison')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

# Add values on top of bars
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        if height > 0:  # Only label bars with non-zero height
            ax.annotate(f'{int(height)}',
                        xy=(rect.get_x() + rect.get_width()/2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

plt.tight_layout()
plt.savefig('algorithm_cost_components_comparison.png')
plt.show()

# Plot 3: Performance comparison line chart
# Calculate normalized costs (lower is better)
max_cost = max(lru_total, lfu_total, online_total)
lru_normalized = lru_total / max_cost * 100
lfu_normalized = lfu_total / max_cost * 100
online_normalized = online_total / max_cost * 100

# Create a radar chart for algorithm performance comparison
labels = ['Total Cost', 'Transmission', 'Cache Miss', 'Fetch Cost', 'Special Features']
lru_radar = [lru_normalized, 
             lru_transmission/max(lru_transmission, lfu_transmission, online_transmission)*100 if max(lru_transmission, lfu_transmission, online_transmission) > 0 else 0,
             lru_cache_miss/max(lru_cache_miss, lfu_cache_miss, online_cache_miss)*100 if max(lru_cache_miss, lfu_cache_miss, online_cache_miss) > 0 else 0,
             lru_fetch/max(lru_fetch, lfu_fetch, online_fetch)*100 if max(lru_fetch, lfu_fetch, online_fetch) > 0 else 0,
             0]  # No special features

lfu_radar = [lfu_normalized,
             lfu_transmission/max(lru_transmission, lfu_transmission, online_transmission)*100 if max(lru_transmission, lfu_transmission, online_transmission) > 0 else 0,
             lfu_cache_miss/max(lru_cache_miss, lfu_cache_miss, online_cache_miss)*100 if max(lru_cache_miss, lfu_cache_miss, online_cache_miss) > 0 else 0,
             lfu_fetch/max(lru_fetch, lfu_fetch, online_fetch)*100 if max(lru_fetch, lfu_fetch, online_fetch) > 0 else 0,
             0]  # No special features

online_radar = [online_normalized,
                online_transmission/max(lru_transmission, lfu_transmission, online_transmission)*100 if max(lru_transmission, lfu_transmission, online_transmission) > 0 else 0,
                online_cache_miss/max(lru_cache_miss, lfu_cache_miss, online_cache_miss)*100 if max(lru_cache_miss, lfu_cache_miss, online_cache_miss) > 0 else 0,
                online_fetch/max(lru_fetch, lfu_fetch, online_fetch)*100 if max(lru_fetch, lfu_fetch, online_fetch) > 0 else 0,
                100]  # Has special features (DIBR, ISL)

# Compare cache hit rates (if available from the simulation)
# Or just use a simple metric like normalized cost (where lower is better)
algs = ['LRU', 'LFU', 'Online Algorithm']
performance = [100 - lru_normalized, 100 - lfu_normalized, 100 - online_normalized]  # Invert so higher is better

plt.figure(figsize=(10, 6))
plt.bar(algs, performance, color=['blue', 'red', 'green'])
plt.title('Algorithm Performance Comparison (Higher is Better)')
plt.ylabel('Performance Score')
plt.ylim(0, 100)
for i, v in enumerate(performance):
    plt.text(i, v + 2, f'{v:.1f}%', ha='center')
plt.tight_layout()
plt.savefig('algorithm_performance_comparison.png')
plt.show()

# Print final comparison
print("\n===== Final Cost Comparison =====")
print(f"LRU Total Cost: {lru_total}")
print(f"LFU Total Cost: {lfu_total}")
print(f"Online Algorithm Total Cost: {online_total}")

# Determine the best algorithm based on total cost
min_cost = min(lru_total, lfu_total, online_total)
if min_cost == lru_total:
    best_alg = "LRU"
elif min_cost == lfu_total:
    best_alg = "LFU"
else:
    best_alg = "Online Algorithm"

print(f"\nThe {best_alg} performed best in this simulation with a total cost of {min_cost}.")





