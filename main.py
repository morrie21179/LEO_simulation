# -*- coding: utf-8 -*-
import os
import pandas as pd
import matplotlib.pyplot as plt
from container import User, Satellite
import random
from container import GroundStation
import numpy as np
from mpl_toolkits.basemap import Basemap
import numpy as np
from mpl_toolkits.basemap import Basemap

# Define video size per view (in MB)
video_size_per_view_mb = 60

timeslot_length = 10
Total_timeslot = 10

# timeslot_length = 5
# Total_timeslot = 100

def visualize_leo_satellite_movement(satellite_table, ground_station_list, user_list, timeslot_to_visualize=0):
    """
    Visualize LEO satellite positions, ground stations, and users on Earth for a specific timeslot
    
    Args:
        satellite_table: Dictionary of satellite objects
        ground_station_list: List of ground station objects
        user_list: List of user objects
        timeslot_to_visualize: Which timeslot to visualize (default: 0)
    """
    import matplotlib.pyplot as plt
    
    # Create figure
    fig = plt.figure(figsize=(15, 10))
    
    # Create map projection
    m = Basemap(projection='mill', llcrnrlat=-80, urcrnrlat=80,
                llcrnrlon=-180, urcrnrlon=180, resolution='c')
    
    # Draw map features
    m.drawcoastlines(linewidth=0.5)
    m.drawcountries(linewidth=0.5)
    m.fillcontinents(color='lightgray', lake_color='aqua')
    m.drawmapboundary(fill_color='aqua')
    
    # Draw parallels and meridians
    m.drawparallels(np.arange(-80, 81, 20), labels=[1,0,0,0], fontsize=8)
    m.drawmeridians(np.arange(-180, 181, 60), labels=[0,0,0,1], fontsize=8)
    
    # Plot ground stations
    gs_lats = [gs.lat for gs in ground_station_list]
    gs_lons = [gs.lon for gs in ground_station_list]
    gs_x, gs_y = m(gs_lons, gs_lats)
    
    m.scatter(gs_x, gs_y, c='red', s=100, marker='^', 
              label='Ground Stations', edgecolors='black', linewidths=1)
    
    # Plot users
    user_lats = [user.lat for user in user_list]
    user_lons = [user.lon for user in user_list]
    user_x, user_y = m(user_lons, user_lats)
    
    m.scatter(user_x, user_y, c='green', s=30, marker='s', 
              label='Users', edgecolors='darkgreen', linewidths=0.5, alpha=0.7)
    
    # Plot satellites for the specified timeslot
    sat_colors = plt.cm.tab10(np.linspace(0, 1, len(satellite_table)))
    
    for i, (sat_name, sat) in enumerate(satellite_table.items()):
        try:
            # Get satellite position using the get_position method from container.py
            if timeslot_to_visualize < len(sat.lat):
                sat_lat = sat.lat.iloc[timeslot_to_visualize]
                sat_lon = sat.lon.iloc[timeslot_to_visualize]
                
                # Convert to map projection
                sat_x, sat_y = m(sat_lon, sat_lat)
                
                # Plot satellite
                m.scatter(sat_x, sat_y, c=[sat_colors[i]], s=80, marker='o', 
                         edgecolors='black', linewidths=1, alpha=0.8)
                
                # Add satellite label
                plt.annotate(sat_name, (sat_x, sat_y), xytext=(5, 5), 
                            textcoords='offset points', fontsize=6, 
                            color='blue', fontweight='bold')
        except Exception as e:
            print(f"Error plotting satellite {sat_name}: {e}")
            continue
    
    # Add legend
    plt.legend(loc='lower left')
    
    # Add title
    plt.title(f'LEO Satellite Network, Ground Stations, and Users at Timeslot {timeslot_to_visualize}\n'
              f'Satellites: {len(satellite_table)}, Ground Stations: {len(ground_station_list)}, Users: {len(user_list)}', 
              fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return fig

def create_satellite_movement_animation(satellite_table, ground_station_list, user_list, total_timeslots=24):
    """
    Create an animation showing satellite movement over time with ground stations and users
    """
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(15, 10))
    
    # Create map projection
    m = Basemap(projection='mill', llcrnrlat=-80, urcrnrlat=80,
                llcrnrlon=-180, urcrnrlon=180, resolution='c', ax=ax)
    
    # Draw static map features
    m.drawcoastlines(linewidth=0.5)
    m.drawcountries(linewidth=0.5)
    m.fillcontinents(color='lightgray', lake_color='aqua')
    m.drawmapboundary(fill_color='aqua')
    m.drawparallels(np.arange(-80, 81, 20), labels=[1,0,0,0], fontsize=8)
    m.drawmeridians(np.arange(-180, 181, 60), labels=[0,0,0,1], fontsize=8)
    
    # Plot ground stations (static)
    gs_lats = [gs.lat for gs in ground_station_list]
    gs_lons = [gs.lon for gs in ground_station_list]
    gs_x, gs_y = m(gs_lons, gs_lats)
    m.scatter(gs_x, gs_y, c='red', s=100, marker='^', 
              label='Ground Stations', edgecolors='black', linewidths=1)
    
    # Plot users (static)
    user_lats = [user.lat for user in user_list]
    user_lons = [user.lon for user in user_list]
    user_x, user_y = m(user_lons, user_lats)
    m.scatter(user_x, user_y, c='green', s=30, marker='s', 
              label='Users', edgecolors='darkgreen', linewidths=0.5, alpha=0.7)
    
    # Initialize satellite scatter plot
    sat_colors = plt.cm.tab10(np.linspace(0, 1, len(satellite_table)))
    satellite_scatters = {}
    
    # Create individual scatter plots for each satellite
    for i, sat_name in enumerate(satellite_table.keys()):
        scatter = ax.scatter([], [], s=120, marker='o', c=[sat_colors[i]], 
                           edgecolors='black', linewidths=2, alpha=0.9, 
                           label=f'Sat-{i+1}' if i < 5 else "")
        satellite_scatters[sat_name] = scatter
    
    # Title text
    title_text = ax.text(0.5, 1.02, '', transform=ax.transAxes, 
                        ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    def animate(frame):
        valid_satellites = 0
        
        for sat_name, sat in satellite_table.items():
            try:
                # Use the satellite data directly from pandas DataFrame
                if frame < len(sat.lat):
                    sat_lat = sat.lat.iloc[frame]
                    sat_lon = sat.lon.iloc[frame]
                    
                    # Validate coordinates
                    if pd.notna(sat_lat) and pd.notna(sat_lon) and \
                       (-90 <= sat_lat <= 90) and (-180 <= sat_lon <= 180):
                        
                        # Convert to map projection
                        sat_x, sat_y = m(sat_lon, sat_lat)
                        
                        # Update satellite position
                        if sat_name in satellite_scatters:
                            satellite_scatters[sat_name].set_offsets([[sat_x, sat_y]])
                            valid_satellites += 1
                    else:
                        # Clear invalid position
                        if sat_name in satellite_scatters:
                            satellite_scatters[sat_name].set_offsets(np.empty((0, 2)))
                else:
                    # Frame beyond data range
                    if sat_name in satellite_scatters:
                        satellite_scatters[sat_name].set_offsets(np.empty((0, 2)))
                        
            except Exception as e:
                if frame == 0:
                    print(f"Error animating satellite {sat_name}: {e}")
                if sat_name in satellite_scatters:
                    satellite_scatters[sat_name].set_offsets(np.empty((0, 2)))
        
        # Update title
        title_text.set_text(f'LEO Satellite Movement - Timeslot {frame}/{total_timeslots-1}\n'
                           f'Valid Satellites: {valid_satellites}/{len(satellite_table)}, '
                           f'Ground Stations: {len(ground_station_list)}, Users: {len(user_list)}')
        
        return list(satellite_scatters.values()) + [title_text]
    
    # Create animation
    anim = animation.FuncAnimation(fig, animate, frames=total_timeslots, 
                                  interval=1000, blit=False, repeat=True)
    
    # Add legend
    ax.legend(loc='lower left', fontsize=8, ncol=2)
    plt.tight_layout()
    
    # Save animation
    try:
        print("Saving animation... This may take a moment.")
        anim.save('satellite_movement_animation.gif', writer='pillow', fps=1, dpi=100)
        print("Animation saved as 'satellite_movement_animation.gif'")
    except Exception as e:
        print(f"Could not save animation: {e}")
    
    return fig, anim

def initialize_random_cache(satellite, total_views, random_seed=None):
    """
    Initialize satellite cache with random multi-view video content
    
    Args:
        satellite: Satellite object to initialize
        total_views: Total number of views available in the system
        random_seed: Optional random seed for reproducible results
    """
    if random_seed is not None:
        random.seed(random_seed + hash(satellite.sat_name) % 1000)
    
    # Clear existing cache
    satellite.cache_state.clear()
    
    # Randomly select views to cache based on storage constraint
    available_views = list(range(total_views))
    random.shuffle(available_views)
    
    # Cache random views up to storage constraint
    views_to_cache = available_views[:satellite.storage_constraint_Z]
    
    for view_id in views_to_cache:
        satellite.cache_view(view_id)
    
    # print(f'Satellite {satellite.sat_name}: Randomly initialized cache with {len(satellite.cache_state)} views')
    # print(f'  Cached views: {sorted(list(satellite.cache_state))[:10]}{"..." if len(satellite.cache_state) > 10 else ""}')

def cache_content_with_eviction(satellite, content_id, timeslot):
    """
    Cache content with proper storage constraint checking and popularity-based eviction
    """
    # Check if content is already cached
    if satellite.is_view_cached(content_id):
        # Update access time and frequency
        satellite.last_access_time[content_id] = timeslot
        satellite.access_frequency[content_id] = satellite.access_frequency.get(content_id, 0) + 1
        return True
    
    # Check storage constraint
    if len(satellite.cache_state) < satellite.storage_constraint_Z:
        # Cache has space, add directly
        satellite.cache_view(content_id)
        satellite.last_access_time[content_id] = timeslot
        satellite.access_frequency[content_id] = satellite.access_frequency.get(content_id, 0) + 1
        return True
    else:
        # Cache is full, need to evict based on popularity
        # Calculate popularity score combining view count and recency
        popularity_scores = {}
        
        for view in satellite.cache_state:
            # Popularity based on cumulative view counts
            view_count_score = satellite.cumulative_view_counts.get(view, 0)
            
            # Recency score (more recent = higher score)
            last_access = satellite.last_access_time.get(view, 0)
            recency_score = 1.0 / (timeslot - last_access + 1)  # Avoid division by zero
            
            # Frequency score
            frequency_score = satellite.access_frequency.get(view, 0)
            
            # Combined popularity score (weighted combination)
            popularity_scores[view] = (0.5 * view_count_score + 
                                     0.3 * frequency_score + 
                                     0.2 * recency_score)
        
        # Find view with minimum popularity to evict
        if popularity_scores:
            min_popularity_view = min(popularity_scores, key=popularity_scores.get)
            min_popularity = popularity_scores[min_popularity_view]
            
            # Calculate new content popularity
            new_content_popularity = (0.5 * satellite.cumulative_view_counts.get(content_id, 0) + 
                                    0.3 * satellite.access_frequency.get(content_id, 0) + 
                                    0.2 * 1.0)  # New content gets maximum recency
            
            # Only evict if new content is more popular
            if new_content_popularity > min_popularity:
                # Evict least popular view
                satellite.evict_view(min_popularity_view)
                
                # Clean up tracking data for evicted view
                if min_popularity_view in satellite.last_access_time:
                    del satellite.last_access_time[min_popularity_view]
                
                # Add new content
                satellite.cache_view(content_id)
                satellite.last_access_time[content_id] = timeslot
                satellite.access_frequency[content_id] = satellite.access_frequency.get(content_id, 0) + 1
                
                return True
            else:
                # New content is not popular enough to replace existing content
                return False
        else:
            # Fallback: evict a random view if no popularity data
            evict_view = next(iter(satellite.cache_state))
            satellite.evict_view(evict_view)
            satellite.cache_view(content_id)
            satellite.last_access_time[content_id] = timeslot
            satellite.access_frequency[content_id] = satellite.access_frequency.get(content_id, 0) + 1
            return True

################################################## User ##########################################################
# load user location from users.csv
users = pd.read_csv('data/users.csv')

# create a list of User objects with video size
user_list = []
for i in range(len(users)):
    user = User(users['id'][i], users['lat'][i], users['lon'][i], 
                users['x'][i], users['y'][i], users['z'][i], 
                video_size_mb=video_size_per_view_mb)
    user_list.append(user)

# sort the user list
user_list.sort()

################################################## Satellite ##########################################################
# load satellite position
satellite_table = {}
total_views = 40000  # Total number of views in multiview video system
storage_constraint_Z = 17500  # Storage constraint in number of views

for file in os.listdir('data/starlink115/satellite_trace'):
    if file.endswith('.csv'):
        satellite_data = pd.read_csv(f'data/starlink115/satellite_trace/{file}')
        satellite_name = file.split('_')[0]
        # Initialize satellite with proper storage constraint and video size
        sat = Satellite(satellite_name, satellite_data, 
                       storage_constraint_Z=storage_constraint_Z, 
                       total_views=total_views, 
                       view_size_mb=video_size_per_view_mb)
        satellite_table[satellite_name] = sat

# Initialize cache state for each satellite with random content
print("Initializing satellite cache states with random multi-view videos...")
total_content_items = total_views  # Use total_views instead of hardcoded value

# Set random seed for reproducible results (optional)
random_seed = 42  # Change this value or set to None for different random initialization

for sat_name, sat in satellite_table.items():
    # Initialize additional attributes needed for caching algorithm
    sat.view_popularity = {}  # Track view popularity P(v)
    sat.neighbor_caches = set()  # Simulated neighbor cache contents
    sat.cumulative_view_counts = {}  # Track cumulative view counts across all timeslots
    sat.access_frequency = {}  # Track access frequency for better popularity estimation
    sat.last_access_time = {}  # Track when each view was last accessed
    
    # Initialize cache with random multi-view videos
    initialize_random_cache(sat, total_views, random_seed)
    
    # Initialize tracking data for randomly cached views
    for view_id in sat.cache_state:
        sat.cumulative_view_counts[view_id] = 1  # Initial count
        sat.access_frequency[view_id] = 1  # Initial frequency
        sat.last_access_time[view_id] = 0  # Initial access time
        sat.view_popularity[view_id] = 1.0 / len(sat.cache_state)  # Initial equal popularity
    
    print(f'Satellite {sat_name}: initialized with {len(sat.cache_state)} random views, '
          f'storage constraint: {sat.storage_constraint_Z}, '
          f'storage used: {sat.get_storage_used_mb():.1f}MB/{sat.get_storage_capacity_mb():.1f}MB')

# Initialize cost tracking
satellite_costs = {sat_name: 0 for sat_name in satellite_table.keys()}
timeslot_costs = []
cache_hit_stats = {sat_name: {'hits': 0, 'misses': 0} for sat_name in satellite_table.keys()}

# # Print cache distribution statistics
# print("\nRandom Cache Distribution Analysis:")
# all_cached_views = set()
# for sat in satellite_table.values():
#     all_cached_views.update(sat.cache_state)

# print(f"Total unique views cached across all satellites: {len(all_cached_views)}")
# print(f"Cache overlap analysis:")

# Calculate cache overlaps between satellites
sat_names = list(satellite_table.keys())
# for i, sat1_name in enumerate(sat_names):
#     for j, sat2_name in enumerate(sat_names[i+1:], i+1):
#         sat1_cache = satellite_table[sat1_name].cache_state
#         sat2_cache = satellite_table[sat2_name].cache_state
#         overlap = len(sat1_cache.intersection(sat2_cache))
#         print(f"  {sat1_name} ??? {sat2_name}: {overlap} common views "
#               f"({overlap/min(len(sat1_cache), len(sat2_cache))*100:.1f}% overlap)")

################################## Ground Stations ##########################################################
# Generate ground station data (since no dataset provided)
print("\nGenerating ground station data...")
ground_stations_data = []
ground_station_locations = [
    {"id": "GS001", "name": "New York", "lat": 40.7128, "lon": -74.0060},
    {"id": "GS002", "name": "Los Angeles", "lat": 34.0522, "lon": -118.2437},
    {"id": "GS003", "name": "London", "lat": 51.5074, "lon": -0.1278},
    {"id": "GS004", "name": "Tokyo", "lat": 35.6762, "lon": 139.6503},
    {"id": "GS005", "name": "Sydney", "lat": -33.8688, "lon": 151.2093},
    {"id": "GS006", "name": "Paris", "lat": 48.8566, "lon": 2.3522},
    {"id": "GS007", "name": "Berlin", "lat": 52.5200, "lon": 13.4050},
    {"id": "GS008", "name": "Rome", "lat": 41.9028, "lon": 12.4964},
    {"id": "GS009", "name": "Madrid", "lat": 40.4168, "lon": -3.7038},
    {"id": "GS010", "name": "Cairo", "lat": 30.0444, "lon": 31.2357},
    {"id": "GS011", "name": "Lagos", "lat": 6.5244, "lon": 3.3792},
    {"id": "GS012", "name": "Johannesburg", "lat": -26.2041, "lon": 28.0473},
    {"id": "GS013", "name": "Nairobi", "lat": -1.2921, "lon": 36.8219},
    {"id": "GS014", "name": "Casablanca", "lat": 33.5731, "lon": -7.5898},
    {"id": "GS015", "name": "Stockholm", "lat": 59.3293, "lon": 18.0686},
    {"id": "GS016", "name": "Oslo", "lat": 59.9139, "lon": 10.7522},
    {"id": "GS017", "name": "Vienna", "lat": 48.2082, "lon": 16.3738},
    {"id": "GS018", "name": "Addis Ababa", "lat": 9.1450, "lon": 40.4897},
    {"id": "GS019", "name": "Amsterdam", "lat": 52.3676, "lon": 4.9041},
    {"id": "GS020", "name": "Tunis", "lat": 36.8065, "lon": 10.1815}
]

# Create ground station CSV data
ground_stations_df = pd.DataFrame(ground_station_locations)
ground_stations_df.to_csv('data/ground_stations.csv', index=False)

# Load ground stations from generated data
ground_stations = pd.read_csv('data/ground_stations.csv')

# Create ground station objects with video size
ground_station_list = []
for i in range(len(ground_stations)):
    gs = GroundStation(
        ground_stations['id'][i], 
        ground_stations['name'][i],
        ground_stations['lat'][i], 
        ground_stations['lon'][i],
        total_views=total_views,  # Total content items in the system
        view_size_mb=video_size_per_view_mb
    )
    ground_station_list.append(gs)

print(f"Created {len(ground_station_list)} ground stations")

########################################################################################################
################################## Online Algorithm (Phase 1) ##########################################
########################################################################################################

# LEO Satellite Cooperative Caching Algorithm Implementation
for i in range(Total_timeslot):
    print(f'===========Time slot {i:03d}===========')
    
    # Reset all user connections at the start of each time slot
    for user in user_list:
        user.sat = None
        user.elevation = 0
    
    # Reset all satellite serving users
    for sat in satellite_table.values():
        sat.serving_users = []
    
    # Find best satellite connection for each user
    for user in user_list:
        best_sat = None
        best_elevation = 0
        
        for sat in satellite_table.values():
            connected, angle = sat.connect_user(i, user)
            if connected and angle > best_elevation:
                best_elevation = angle
                best_sat = sat
        
        # Connect user to best satellite
        if best_sat:
            user.elevation = best_elevation
            user.sat = best_sat
            best_sat.serving_users.append(user.user_id)

    # Update neighbor cache information BEFORE processing requests
    sat_names = list(satellite_table.keys())
    for sat in satellite_table.values():
        current_idx = sat_names.index(sat.sat_name)
        
        # Get neighbor caches (previous and next satellite)
        neighbor_cache_union = set()
        if current_idx > 0:
            prev_sat = satellite_table[sat_names[current_idx - 1]]
            neighbor_cache_union.update(prev_sat.cache_state)
        if current_idx < len(sat_names) - 1:
            next_sat = satellite_table[sat_names[current_idx + 1]]
            neighbor_cache_union.update(next_sat.cache_state)
        
        sat.neighbor_caches = neighbor_cache_union

    # Simulate content requests and cooperative caching decisions
    timeslot_total_cost = 0
    
    for sat in satellite_table.values():
        sat_cost = 0
        
        # Update request range R_n(t) and number of viewers N_n(t)
        request_range = []
        
        # Each connected user generates multiview video request
        for user_id in sat.serving_users:
            # Find the user object
            user = next(u for u in user_list if u.user_id == user_id)
            
            # Generate multiview video request using User methods
            view_index = random.randint(1, total_views)
            view_range_B = 3
            D = 3
            
            success, result = user.generate_request(view_index, view_range_B)
            
            if success:
                # Extract request range from the generated request
                half_range = view_range_B // 2
                start_view = view_index - half_range
                end_view = view_index + half_range
                
                # Add all views in the range to request_range
                for view in range(start_view, end_view + 1):
                    if 0 <= view < total_views:  # Only valid view indices
                        request_range.append(view)
                        # Update cumulative view counts
                        sat.cumulative_view_counts[view] = sat.cumulative_view_counts.get(view, 0) + 1
        
        # Calculate possibility of view v as P(v) using cumulative view counts
        total_cumulative_views = sum(sat.cumulative_view_counts.values())
        if total_cumulative_views > 0:
            for view in sat.cumulative_view_counts.keys():
                sat.view_popularity[view] = sat.cumulative_view_counts[view] / total_cumulative_views
        
        # Find nearest ground station for this satellite
        nearest_gs = None
        min_distance = float('inf')

        # Get satellite position for current timeslot
        if i < len(sat.lat):
            sat_lat = sat.lat.iloc[i]
            sat_lon = sat.lon.iloc[i]
            sat_alt = sat.alt.iloc[i]
            
            for gs in ground_station_list:
                distance = gs.calculate_distance_to_satellite(sat_lat, sat_lon, sat_alt)
                if distance < min_distance:
                    min_distance = distance
                    nearest_gs = gs
        
        # Connect satellite to nearest ground station
        if nearest_gs:
            nearest_gs.connect_satellite(sat)
        
        ############################## Online Algorithm (Phase 1) ##############################
        ########################################################################################
        # # Process each request in the range
        # for requested_content in request_range:
        #     # Calculate transmission cost £n_j based on LEO cooperative caching algorithm
        #     if sat.is_view_cached(requested_content):
        #         # Case 1: Content is in local cache V_n(t)
        #         tau_j = 1  # c_{n,s}(z_j,d_j) - local serving cost
        #         cache_hit_stats[sat.sat_name]['hits'] += 1
        #         sat_cost += tau_j
                
        #         # Update access statistics
        #         sat.last_access_time[requested_content] = i
        #         sat.access_frequency[requested_content] = sat.access_frequency.get(requested_content, 0) + 1
                
        #     elif requested_content in sat.neighbor_caches:
        #         # Case 2: Content is in neighbor cache (V_{n-1}(t) ¡å V_{n+1}(t)) \ V_n(t)
        #         tau_j = 3 + 1  # c_{ISL}(z_j,d^{ISL}_j) + c_{n,s}(z_j,d_j)
        #         cache_hit_stats[sat.sat_name]['hits'] += 1
        #         sat_cost += tau_j

        #         # Cooperative caching decision: fetch from neighbor with proper eviction
        #         cache_content_with_eviction(sat, requested_content, i)
                
        #     else:
        #         # Case 3: Content not in local or neighbor cache - fetch from ground station
        #         if nearest_gs and nearest_gs.has_view(requested_content):
        #             # Use ground station to transmit content to satellite
        #             transmitted_views = nearest_gs.transmit_to_satellite(sat, [requested_content])
                    
        #             # Calculate transmission cost using ground station method
        #             data_size_mb = len(transmitted_views) * video_size_per_view_mb
        #             tau_j = nearest_gs.calculate_transmission_cost(min_distance, data_size_mb) + 1
        #         else:
        #             # Fallback cost if no ground station available or content not available
        #             tau_j = 50  # High cost for unavailable content
                
        #         cache_hit_stats[sat.sat_name]['misses'] += 1
        #         sat_cost += tau_j
                
        #         # Caching decision with proper storage constraint and eviction
        #         cache_content_with_eviction(sat, requested_content, i)

        #     # Dynamic Programming - Calculate £g_{h,j}
        #     k = max(requested_content, D)
        #     mu_values = {}

        #     # Calculate £g_{h,j} for h from max(j-D, h) to j ? i ? j
        #     for idx in range(k, requested_content):
        #         # Calculate the minimum cost for serving content h at timeslot j
        #         # £g_{h,j} = min_{max(j-D,h) ? i ? j} (£n_j + £g_{h,i} + (£\(j-i) + T_{DIBR})(j-i-1))
        #         min_cost = float('inf')

        #         # DIBR processing cost: (£\(j-i) + T_DIBR)(j-i-1)
        #         alpha = 0.1  # Processing cost factor
        #         T_DIBR = 2   # Base DIBR processing time
        #         j_i = requested_content - idx
        #         dibr_cost = (alpha * j_i + T_DIBR) * max(j_i - 1, 0)

        #         # Total cost for this choice of i
        #         total_cost = tau_j 

        #         if total_cost < min_cost:
        #             min_cost = total_cost

        #         # # Store the minimum cost
        #         # mu_values[(h, requested_content)] = min_cost
            
        #     # The optimal cost for serving this content is £g_{requested_content, requested_content}
        #     optimal_cost = mu_values.get((requested_content, requested_content), tau_j)
            
        #     # Update the satellite cost with the optimal cost instead of just tau_j
        #     sat_cost = sat_cost - tau_j + optimal_cost  # Replace the previously added tau_j
        if request_range:
            h = min(request_range)
            l = max(request_range)
            alpha = 0.1
            T_DIBR = 2
            D = 3

            mu = dict()
            prev = dict()
            tau_dict = dict()

            def get_tau_j(view_j):
                if sat.is_view_cached(view_j):
                    return 1
                elif view_j in sat.neighbor_caches:
                    return 4
                elif nearest_gs and nearest_gs.has_view(view_j):
                    # return nearest_gs.calculate_transmission_cost(min_distance, video_size_per_view_mb) + 1
                    return 20
                else:
                    return 40

            # DP initial condition
            tau_h = get_tau_j(h)
            mu[h] = tau_h
            prev[h] = None
            tau_dict[h] = tau_h

            # Fill DP table for [h+1, l]
            for j in range(h+1, l+1):
                tau_j = get_tau_j(j)
                tau_dict[j] = tau_j
                min_cost = float('inf')
                best_i = None
                for i in range(max(j-D, h), j):
                    dibr_cost = (alpha * (j - i) + T_DIBR) * (j - i - 1)
                    cost = tau_j + mu[i] + dibr_cost
                    if cost < min_cost:
                        min_cost = cost
                        best_i = i
                mu[j] = min_cost
                prev[j] = best_i

            # Backtrack to get transfer points (fetch points)
            transfer_points = []
            j = l
            while j is not None:
                transfer_points.append(j)
                j = prev[j]
            transfer_points = transfer_points[::-1]

            # Decide sets for fetch, DIBR, ISL, etc.
            V_fetch = set(transfer_points)
            V_DIBR = set()
            V_ISL = set()
            V_trans = set()
            V_evicted = set()

            # Classify each subinterval
            for idx in range(len(transfer_points)-1):
                start = transfer_points[idx]
                end = transfer_points[idx+1]
                # If end > start+1, then views in between are synthesized by DIBR
                if end > start+1:
                    for synth in range(start+1, end):
                        V_DIBR.add(synth)

            # For each fetch point, classify ISL/local/GS
            for v in V_fetch:
                if sat.is_view_cached(v):
                    V_trans.add(v)  # Already cached, just transmit
                    cache_hit_stats[sat.sat_name]['hits'] += 1
                    sat.last_access_time[v] = i
                    sat.access_frequency[v] = sat.access_frequency.get(v, 0) + 1
                elif v in sat.neighbor_caches:
                    V_ISL.add(v)
                    cache_hit_stats[sat.sat_name]['hits'] += 1
                    cache_content_with_eviction(sat, v, i)
                else:
                    # Fetch from ground station (or fallback)
                    cache_hit_stats[sat.sat_name]['misses'] += 1
                    if nearest_gs and nearest_gs.has_view(v):
                        nearest_gs.transmit_to_satellite(sat, [v])
                        cache_content_with_eviction(sat, v, i)
                    # If needed, track evicted views here

            # Add the DP minimum total cost
            sat_cost += mu[l]       

        # Add to satellite's total cost
        satellite_costs[sat.sat_name] += sat_cost
        timeslot_total_cost += sat_cost
        
        # Calculate cache hit rate
        total_requests = cache_hit_stats[sat.sat_name]['hits'] + cache_hit_stats[sat.sat_name]['misses']
        hit_rate = cache_hit_stats[sat.sat_name]['hits'] / total_requests if total_requests > 0 else 0
        
        # Calculate cache utilization
        cache_utilization = sat.get_cache_utilization()
        
        # Disconnect satellite from ground station
        if nearest_gs:
            nearest_gs.disconnect_satellite(sat)
        
        nearest_gs_name = nearest_gs.name if nearest_gs else "None"
        # print(f'Satellite {sat.sat_name}: serving {len(sat.serving_users)} users, '
        #       f'cost: {sat_cost}, cache size: {len(sat.cache_state)}/{sat.storage_constraint_Z}, '
        #       f'utilization: {cache_utilization:.2f}, hit rate: {hit_rate:.2f}, '
        #       f'storage: {sat.get_storage_used_mb():.1f}MB/{sat.get_storage_capacity_mb():.1f}MB, '
        #       f'neighbor cache size: {len(sat.neighbor_caches)}, nearest GS: {nearest_gs_name}')
    
    timeslot_costs.append(timeslot_total_cost)
    print(f'Total served users: {sum(len(sat.serving_users) for sat in satellite_table.values())}')
    print(f'Total cost this timeslot: {timeslot_total_cost}')
    print()

########################################################################################################

# Print final cost and performance summary
print("="*50)
print("FINAL PERFORMANCE SUMMARY - LEO COOPERATIVE CACHING WITH RANDOM INITIALIZATION")
print("="*50)
total_system_cost = sum(satellite_costs.values())

for sat_name, cost in sorted(satellite_costs.items()):
    total_requests = cache_hit_stats[sat_name]['hits'] + cache_hit_stats[sat_name]['misses']
    hit_rate = cache_hit_stats[sat_name]['hits'] / total_requests if total_requests > 0 else 0
    sat = satellite_table[sat_name]
    total_cumulative_views = sum(sat.cumulative_view_counts.values())
    cache_utilization = sat.get_cache_utilization()
    print(f'Satellite {sat_name}: Total cost = {cost}, Hit rate = {hit_rate:.3f}, '
          f'Total requests = {total_requests}, Cumulative views = {total_cumulative_views}, '
          f'Cache utilization = {cache_utilization:.3f} ({len(sat.cache_state)}/{sat.storage_constraint_Z}), '
          f'Storage: {sat.get_storage_used_mb():.1f}MB/{sat.get_storage_capacity_mb():.1f}MB')

print(f'\nGround Stations:')
for gs in ground_station_list:
    stats = gs.get_transmission_statistics()
    total_storage_gb = (gs.total_views * gs.view_size_mb) / 1024
    print(f'  {gs.station_id}: {gs.name} at ({gs.lat:.2f}, {gs.lon:.2f}) - '
          f'Transmissions: {stats["total_transmissions"]}, Views sent: {stats["total_views_transmitted"]}, '
          f'Data transmitted: {stats["total_data_transmitted_gb"]:.2f}GB, Total storage: {total_storage_gb:.1f}GB')

print(f'\nTotal system cost: {total_system_cost}')
overall_hits = sum(stats['hits'] for stats in cache_hit_stats.values())
overall_requests = sum(stats['hits'] + stats['misses'] for stats in cache_hit_stats.values())
overall_hit_rate = overall_hits / overall_requests if overall_requests > 0 else 0
print(f'Overall cache hit rate: {overall_hit_rate:.3f}')

# System storage statistics
total_system_storage_used_mb = sum(sat.get_storage_used_mb() for sat in satellite_table.values())
total_system_storage_capacity_mb = sum(sat.get_storage_capacity_mb() for sat in satellite_table.values())
print(f'Total system storage: {total_system_storage_used_mb:.1f}MB/{total_system_storage_capacity_mb:.1f}MB '
      f'({total_system_storage_used_mb/total_system_storage_capacity_mb*100:.1f}% utilized)')

# # Random initialization summary
# print(f'\nRandom Cache Initialization Summary:')
# print(f'Random seed used: {random_seed}')
# print(f'Total unique views cached: {len(all_cached_views)} out of {total_views} ({len(all_cached_views)/total_views*100:.1f}%)')

# Create plotting data
sat_names = list(satellite_table.keys())
sat_costs = [satellite_costs[sat_name] for sat_name in sat_names]

# Plot 1: Total cost per satellite (bar chart)
plt.figure(figsize=(12, 6))
plt.bar(range(len(sat_names)), sat_costs, color='skyblue', edgecolor='navy', alpha=0.7)
plt.xlabel('Satellite')
plt.ylabel('Total Cost')
plt.title('Total Cost per Satellite (Random Cache Initialization)')
plt.xticks(range(len(sat_names)), sat_names, rotation=45, ha='right')
plt.grid(True, alpha=0.3)

# Add value labels on bars
for i, cost in enumerate(sat_costs):
    plt.text(i, cost + max(sat_costs) * 0.01, str(cost), 
             ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('total_cost_per_satellite.png', dpi=300, bbox_inches='tight')
plt.show()

# Plot 2: Cost over time (line chart)
plt.figure(figsize=(12, 6))
plt.plot(range(Total_timeslot), timeslot_costs, marker='o', linewidth=2, 
         markersize=6, color='red', markerfacecolor='orange')
plt.xlabel('Timeslot')
plt.ylabel('Total Cost')
plt.title('Total System Cost per Timeslot (Random Cache Initialization)')
plt.grid(True, alpha=0.3)
plt.xlim(-0.5, Total_timeslot - 0.5)
plt.tight_layout()
plt.savefig('cost_over_time.png', dpi=300, bbox_inches='tight')
plt.show()

# Plot 3: Cache hit rates per satellite
hit_rates = [cache_hit_stats[sat]['hits'] / (cache_hit_stats[sat]['hits'] + cache_hit_stats[sat]['misses']) 
             if (cache_hit_stats[sat]['hits'] + cache_hit_stats[sat]['misses']) > 0 else 0 
             for sat in sat_names]

plt.figure(figsize=(12, 6))
plt.bar(range(len(sat_names)), hit_rates, color='lightgreen', edgecolor='darkgreen', alpha=0.7)
plt.xlabel('Satellite')
plt.ylabel('Cache Hit Rate')
plt.title('Cache Hit Rate per Satellite (Random Cache Initialization)')
plt.xticks(range(len(sat_names)), sat_names, rotation=45, ha='right')
plt.grid(True, alpha=0.3)
plt.ylim(0, 1)

# Add value labels on bars
for i, rate in enumerate(hit_rates):
    plt.text(i, rate + 0.01, f'{rate:.3f}', 
             ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('cache_hit_rates_per_satellite.png', dpi=300, bbox_inches='tight')
plt.show()

# Additional statistics
print(f'\nAverage cost per timeslot: {total_system_cost / Total_timeslot:.2f}')
print(f'Peak timeslot cost: {max(timeslot_costs)}')
print(f'Minimum timeslot cost: {min(timeslot_costs)}')

print(f'\nRandom Cache Initialization Performance Analysis:')
print(f'Total ground stations available: {len(ground_station_list)}')
print(f'Cache initialization: Random selection of multi-view videos')
print(f'Cache miss resolution: Ground station-based content delivery')


# Visualize satellite positions, ground stations, and users
print("\nGenerating satellite network visualization...")
visualization_timeslot = 0  # You can change this to visualize different timeslots
fig = visualize_leo_satellite_movement(satellite_table, ground_station_list, user_list, visualization_timeslot)
plt.show()

# Optionally create animation of satellite movement with users
print("\nCreating satellite movement animation with users...")
fig_anim, anim = create_satellite_movement_animation(satellite_table, ground_station_list, user_list, Total_timeslot)
plt.show()
