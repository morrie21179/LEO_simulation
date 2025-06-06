import numpy as np
import pandas as pd

EARTH_RADIUS = 6371

# create a class to represent a user
class User:
    def __init__(self, user_id, lat, lon, x, y, z, video_size_mb=10):
        self.user_id = user_id
        self.lat = lat
        self.lon = lon
        self.x = x
        self.y = y
        self.z = z
        self.video_size_mb = video_size_mb  # Size per view in MB

        # serving satellite information
        self.elevation = 0
        self.sat = None

    def __str__(self):
        return f'User {self.user_id} is at ({self.lat}, {self.lon}), video size: {self.video_size_mb}MB per view'
    
    # define the order of users
    def __lt__(self, other):
        return self.user_id < other.user_id
    
    # generate user request for multiview video
    def generate_request(self, view_index=None, view_range_B=3):
        if self.sat is None:
            return False, "No satellite connection available"
        
        # Default values if not provided
        if view_index is None:
            view_index = np.random.randint(0, 1000)  # Random view index
        if view_range_B is None:
            view_range_B = np.random.randint(5, 20)  # Random view range
        
        return self.make_request(self.sat.sat_name, view_index, view_range_B)
    
    # make a request to the serving LEO satellite for multiview video
    def make_request(self, satellite_name, view_index, view_range_B):
        if self.sat is None:
            return False, "No satellite connection available"
        
        # Calculate the request range: [R_n(t) - floor(B/2), R_n(t) + floor(B/2)]
        half_range = view_range_B // 2
        start_view = view_index - half_range
        end_view = view_index + half_range
        
        # Calculate total data size for the request
        total_views = view_range_B
        total_data_size_mb = total_views * self.video_size_mb
        
        request_data = {
            'satellite_name': satellite_name,
            'center_view': view_index,  # R_n(t)
            'view_range': view_range_B,  # B
            'start_view': start_view,
            'end_view': end_view,
            'user_id': self.user_id,
            'video_size_per_view_mb': self.video_size_mb,
            'total_data_size_mb': total_data_size_mb
        }
        
        # Send multiview video request to the serving satellite
        return self.sat.handle_user_request(self.user_id, request_data)
    
    # check if user has an active satellite connection
    def has_satellite_connection(self):
        return self.sat is not None
    
    # calculate bandwidth requirement for the request
    def calculate_bandwidth_requirement(self, view_range_B):
        """Calculate bandwidth requirement in Mbps for a given view range"""
        total_data_mb = view_range_B * self.video_size_mb
        # Assuming real-time streaming requirement (data needs to be delivered within 1 second)
        bandwidth_mbps = total_data_mb * 8  # Convert MB to Mbps
        return bandwidth_mbps

# create a class to represent a satellite which read from each satellite_position.csv
class Satellite:
    def __init__(self, sat_name, sat_csv, storage_constraint_Z=100, total_views=360, view_size_mb=10):
        self.sat_csv = sat_csv
        self.sat_name = sat_name
        self.time = sat_csv['time']
        self.x = sat_csv['x']
        self.y = sat_csv['y']
        self.z = sat_csv['z']
        self.lat = sat_csv['lat']
        self.lon = sat_csv['lon']
        self.alt = sat_csv['alt']
        self.min_elevation = 25
        self.earth_x = EARTH_RADIUS * np.cos(self.lat * np.pi / 180) * np.cos(self.lon * np.pi / 180)
        self.earth_y = EARTH_RADIUS * np.cos(self.lat * np.pi / 180) * np.sin(self.lon * np.pi / 180)
        self.earth_z = EARTH_RADIUS * np.sin(self.lat * np.pi / 180)
        self.serving_users = []
        self.storage_constraint_Z = storage_constraint_Z  # Cache storage constraint in number of views
        self.total_views = total_views  # Total number of views (e.g., 360 for panoramic video)
        self.view_size_mb = view_size_mb  # Size per view in MB
        
        # V_n(t): Cache state - set of views stored in satellite n at time t
        self.cache_state = set()  # V_n(t)
        
        # Request history
        self.request_history = []
        
        # Initialize cache with random views (or can be empty initially)
        self._initialize_cache()

    def __str__(self):
        return f'Satellite {self.sat_name}, serving users {self.serving_users}, cached views: {len(self.cache_state)}, storage: {self.get_storage_used_mb():.1f}MB'

    def __lt__(self, other):
        return self.sat_name < other.sat_name
    
    def _initialize_cache(self):
        """Initialize cache with random views up to storage constraint"""
        if self.storage_constraint_Z > 0:
            initial_cache_size = min(self.storage_constraint_Z, self.total_views // 4)  # Start with 25% capacity
            initial_views = np.random.choice(self.total_views, initial_cache_size, replace=False)
            self.cache_state = set(initial_views)
    
    def get_cache_state(self):
        """Get current cache state V_n(t)"""
        return self.cache_state.copy()
    
    def is_view_cached(self, view_index):
        """Check if a specific view is cached"""
        return view_index in self.cache_state
    
    def cache_view(self, view_index):
        """Add a view to cache if storage permits"""
        if len(self.cache_state) < self.storage_constraint_Z:
            self.cache_state.add(view_index)
            return True
        return False
    
    def evict_view(self, view_index):
        """Remove a view from cache"""
        if view_index in self.cache_state:
            self.cache_state.remove(view_index)
            return True
        return False
    
    def get_cache_utilization(self):
        """Get current cache utilization ratio"""
        return len(self.cache_state) / self.storage_constraint_Z if self.storage_constraint_Z > 0 else 0
    
    def get_storage_used_mb(self):
        """Get current storage used in MB"""
        return len(self.cache_state) * self.view_size_mb
    
    def get_storage_capacity_mb(self):
        """Get total storage capacity in MB"""
        return self.storage_constraint_Z * self.view_size_mb
    
    # get the position of satellite at a specific time
    def get_position(self, time):
        return self.x[time], self.y[time], self.z[time], self.earth_x[time], self.earth_y[time], self.earth_z[time]
    
    # define the connection relationship between satellite and user
    def connect_user(self, time, user):
        ux, uy, uz = user.x, user.y, user.z
        sx, sy, sz, sex, sey, sez = self.get_position(time)

        # calculate theta
        a = np.sqrt((ux - sex) ** 2 + (uy - sey) ** 2 + (uz - sez) ** 2)
        theta = np.arccos(a / (2 * EARTH_RADIUS))

        # calculate phi
        height = self.alt[time]
        b = np.sqrt((ux - sx) ** 2 + (uy - sy) ** 2 + (uz - sz) ** 2)
        if height / b * np.sin(theta) > 1:
            phi = 1
        else:
            phi = np.arcsin(height / b * np.sin(theta))

        return (theta + phi) * 180 / np.pi >= self.min_elevation + 90, (theta + phi) * 180 / np.pi
    
    # handle user request for multiview video
    def handle_user_request(self, user_id, request_data):
        if user_id not in self.serving_users:
            return False, f"User {user_id} is not connected to satellite {self.sat_name}"
        
        center_view = request_data['center_view']
        start_view = request_data['start_view']
        end_view = request_data['end_view']
        video_size_per_view = request_data.get('video_size_per_view_mb', self.view_size_mb)
        total_data_size = request_data.get('total_data_size_mb', 0)
        
        # Generate list of requested views
        requested_views = list(range(start_view, end_view + 1))
        
        # Determine which views are available in cache
        cached_views = [v for v in requested_views if v in self.cache_state]
        missing_views = [v for v in requested_views if v not in self.cache_state]
        
        # Calculate data sizes
        cached_data_size_mb = len(cached_views) * video_size_per_view
        missing_data_size_mb = len(missing_views) * video_size_per_view
        
        # Store request in history
        self.request_history.append({
            'user_id': user_id,
            'requested_views': requested_views,
            'cached_views': cached_views,
            'missing_views': missing_views,
            'cached_data_size_mb': cached_data_size_mb,
            'missing_data_size_mb': missing_data_size_mb,
            'total_data_size_mb': total_data_size,
            'timestamp': len(self.request_history)
        })
        
        print(f"Satellite {self.sat_name} processing request from user {user_id}:")
        print(f"  Center view: {center_view}")
        print(f"  Requested views: {start_view} to {end_view}")
        print(f"  Cached views: {len(cached_views)} ({cached_data_size_mb:.1f}MB)")
        print(f"  Missing views: {len(missing_views)} ({missing_data_size_mb:.1f}MB)")
        
        response = {
            'success': True,
            'cached_views': cached_views,
            'missing_views': missing_views,
            'cached_data_size_mb': cached_data_size_mb,
            'missing_data_size_mb': missing_data_size_mb,
            'satellite': self.sat_name
        }
        
        return True, response


# Ground Station class to serve as the source for multiview video content
class GroundStation:
    def __init__(self, station_id, name, lat, lon, total_views=360, view_size_mb=10):
        self.station_id = station_id
        self.name = name
        self.lat = float(lat)
        self.lon = float(lon)
        self.total_views = int(total_views)
        self.view_size_mb = view_size_mb  # Size per view in MB
        
        # Ground station has all views available (complete multiview video)
        self.available_views = set(range(self.total_views))
        
        # Track requests and transmissions
        self.transmission_history = []
        self.connected_satellites = []
        
    def __str__(self):
        total_size_gb = (len(self.available_views) * self.view_size_mb) / 1024
        return f'Ground Station {self.station_id} ({self.name}) at ({self.lat}, {self.lon}) with {len(self.available_views)} views ({total_size_gb:.1f}GB)'
    
    def has_view(self, view_index):
        """Check if ground station has a specific view (always True)"""
        return view_index in self.available_views
    
    def get_views_range(self, start_view, end_view):
        """Get a range of views from the ground station"""
        requested_views = list(range(start_view, end_view + 1))
        available_requested = [v for v in requested_views if v in self.available_views]
        return available_requested
    
    def transmit_to_satellite(self, satellite, requested_views):
        """Transmit requested views to a satellite"""
        if not isinstance(requested_views, list):
            requested_views = [requested_views]
        
        available_views = [v for v in requested_views if v in self.available_views]
        transmitted_data_size_mb = len(available_views) * self.view_size_mb
        
        # Record transmission
        transmission_record = {
            'satellite': satellite.sat_name,
            'requested_views': requested_views,
            'transmitted_views': available_views,
            'transmitted_data_size_mb': transmitted_data_size_mb,
            'timestamp': len(self.transmission_history)
        }
        self.transmission_history.append(transmission_record)
        
        print(f"Ground Station {self.station_id} transmitting {len(available_views)} views ({transmitted_data_size_mb:.1f}MB) to satellite {satellite.sat_name}")
        
        return available_views
    
    def connect_satellite(self, satellite):
        """Establish connection with a satellite"""
        if satellite.sat_name not in self.connected_satellites:
            self.connected_satellites.append(satellite.sat_name)
    
    def disconnect_satellite(self, satellite):
        """Disconnect from a satellite"""
        if satellite.sat_name in self.connected_satellites:
            self.connected_satellites.remove(satellite.sat_name)
    
    def get_transmission_statistics(self):
        """Get statistics about transmissions from ground station"""
        total_transmissions = len(self.transmission_history)
        total_views_transmitted = sum(len(t['transmitted_views']) for t in self.transmission_history)
        total_data_transmitted_mb = sum(t['transmitted_data_size_mb'] for t in self.transmission_history)
        
        return {
            'station_id': self.station_id,
            'total_transmissions': total_transmissions,
            'total_views_transmitted': total_views_transmitted,
            'total_data_transmitted_mb': total_data_transmitted_mb,
            'total_data_transmitted_gb': total_data_transmitted_mb / 1024,
            'connected_satellites': len(self.connected_satellites)
        }
    
    def calculate_distance_to_satellite(self, sat_lat, sat_lon, sat_alt):
        """Calculate distance between ground station and satellite"""
        gs_lat_rad = np.radians(float(self.lat))
        gs_lon_rad = np.radians(float(self.lon))
        sat_lat_rad = np.radians(float(sat_lat))
        sat_lon_rad = np.radians(float(sat_lon))
        
        # Calculate angular distance using haversine formula
        dlat = sat_lat_rad - gs_lat_rad
        dlon = sat_lon_rad - gs_lon_rad
        a = np.sin(dlat/2)**2 + np.cos(gs_lat_rad) * np.cos(sat_lat_rad) * np.sin(dlon/2)**2
        angular_distance = 2 * np.arcsin(np.sqrt(a))
        
        # Calculate surface distance
        surface_distance = EARTH_RADIUS * angular_distance
        
        # Calculate 3D distance including altitude
        distance = np.sqrt(surface_distance**2 + float(sat_alt)**2)
        
        return distance
    
    def calculate_transmission_cost(self, distance, data_size_mb):
        """Calculate transmission cost based on distance and data size"""
        # Simple cost model: higher cost for longer distances and larger data
        base_cost = 10
        distance_factor = distance / 1000  # Convert to km and scale
        data_factor = data_size_mb / 100  # Scale data size
        return base_cost + distance_factor + data_factor

# System model class to manage the overall LEO caching system
class LEOCachingSystem:
    def __init__(self, satellites, users):
        self.satellites = satellites
        self.users = users
        self.current_time = 0
        
    def update_time(self, time):
        """Update system time and all satellite-user connections"""
        self.current_time = time
        self._update_connections()
    
    def _update_connections(self):
        """Update satellite-user connections based on current time"""
        # Reset all connections
        for user in self.users:
            user.sat = None
            user.elevation = 0
        
        for satellite in self.satellites:
            satellite.serving_users = []
        
        # Establish new connections
        for user in self.users:
            best_satellite = None
            best_elevation = 0
            
            for satellite in self.satellites:
                can_connect, elevation = satellite.connect_user(self.current_time, user)
                if can_connect and elevation > best_elevation:
                    best_satellite = satellite
                    best_elevation = elevation
            
            if best_satellite:
                user.sat = best_satellite
                user.elevation = best_elevation
                best_satellite.serving_users.append(user.user_id)
    
    def get_system_cache_state(self):
        """Get cache state of all satellites V(t) = {V_1(t), V_2(t), ..., V_N(t)}"""
        return {sat.sat_name: sat.get_cache_state() for sat in self.satellites}
    
    def get_system_statistics(self):
        """Get overall system performance statistics"""
        total_cache_utilization = sum(sat.get_cache_utilization() for sat in self.satellites) / len(self.satellites)
        total_served_users = sum(len(sat.serving_users) for sat in self.satellites)
        total_storage_used_mb = sum(sat.get_storage_used_mb() for sat in self.satellites)
        total_storage_capacity_mb = sum(sat.get_storage_capacity_mb() for sat in self.satellites)
        
        return {
            'time': self.current_time,
            'avg_cache_utilization': total_cache_utilization,
            'total_served_users': total_served_users,
            'active_satellites': len([sat for sat in self.satellites if sat.serving_users]),
            'total_storage_used_mb': total_storage_used_mb,
            'total_storage_capacity_mb': total_storage_capacity_mb,
            'total_storage_used_gb': total_storage_used_mb / 1024,
            'total_storage_capacity_gb': total_storage_capacity_mb / 1024
        }
