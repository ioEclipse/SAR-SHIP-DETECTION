"""
AIS Detector Module

This module handles the integration of AIS (Automatic Identification System) data
with SAR ship detection model predictions to identify which ships have and don't 
have AIS transmissions.

Key Components:
1. AIS Data Acquisition and Parsing
2. Spatial-Temporal Matching with SAR Detections
3. Ship Classification (AIS-equipped vs Non-AIS)
4. Data Fusion and Output Generation
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import requests
import asyncio
import websockets
import threading
import time
import os
from urllib.parse import urlencode


@dataclass
class AISRecord:
    """Data structure for AIS transmission records"""
    mmsi: str  # Maritime Mobile Service Identity
    timestamp: datetime
    latitude: float
    longitude: float
    speed: float  # knots
    course: float  # degrees
    vessel_type: Optional[str] = None
    vessel_name: Optional[str] = None
    length: Optional[float] = None
    width: Optional[float] = None


@dataclass
class SARDetection:
    """Data structure for SAR ship detection results"""
    detection_id: str
    latitude: float
    longitude: float
    confidence: float
    bbox: Tuple[float, float, float, float]  # x, y, width, height
    timestamp: datetime
    size_estimate: Optional[float] = None


@dataclass
class MatchedShip:
    """Data structure for matched ship (SAR detection + AIS data)"""
    sar_detection: SARDetection
    ais_record: Optional[AISRecord]
    match_confidence: float
    has_ais: bool
    distance_meters: Optional[float] = None
    time_diff_seconds: Optional[float] = None


def load_config(config_path: str = "config.json") -> Dict:
    """
    Load configuration from JSON file or environment variables.
    
    Args:
        config_path: Path to configuration JSON file
        
    Returns:
        Configuration dictionary
    """
    config = {}
    
    # Try to load from JSON file
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
        except Exception as e:
            print(f"Warning: Could not load config from {config_path}: {e}")
    
    # Override with environment variables if available
    if os.getenv('AISSTREAM_API_KEY'):
        if 'aisstream' not in config:
            config['aisstream'] = {}
        config['aisstream']['api_key'] = os.getenv('AISSTREAM_API_KEY')
    
    return config


class AISDetector:
    """
    Main class for AIS detection and integration with SAR ship detections.
    
    This class handles the complete workflow of:
    1. Loading and parsing AIS data
    2. Matching AIS records with SAR detections
    3. Classifying ships as AIS-equipped or non-AIS
    4. Generating comprehensive output reports
    """
    
    def __init__(self, 
                 spatial_threshold_meters: Optional[float] = None,
                 temporal_threshold_seconds: Optional[float] = None,
                 min_match_confidence: Optional[float] = None,
                 aisstream_api_key: Optional[str] = None,
                 config_path: str = "config.json"):
        """
        Initialize AIS Detector with matching parameters.
        
        Args:
            spatial_threshold_meters: Maximum distance for spatial matching
            temporal_threshold_seconds: Maximum time difference for temporal matching
            min_match_confidence: Minimum confidence score for valid matches
            aisstream_api_key: API key for aisstream.io service
            config_path: Path to configuration file
        """
        # Load configuration
        self.config = load_config(config_path)
        
        # Set parameters from config or use provided values
        ais_config = self.config.get('ais_detector', {})
        self.spatial_threshold = spatial_threshold_meters or ais_config.get('spatial_threshold_meters', 500.0)
        self.temporal_threshold = temporal_threshold_seconds or ais_config.get('temporal_threshold_seconds', 3600.0)
        self.min_match_confidence = min_match_confidence or ais_config.get('min_match_confidence', 0.7)
        
        # Set API key from parameter, config, or environment
        self.aisstream_api_key = (
            aisstream_api_key or 
            self.config.get('aisstream', {}).get('api_key') or
            os.getenv('AISSTREAM_API_KEY')
        )
        
        # Data storage
        self.ais_records: List[AISRecord] = []
        self.sar_detections: List[SARDetection] = []
        self.matched_ships: List[MatchedShip] = []
    
    def load_ais_data(self, ais_data_source: Union[str, Dict], 
                     bbox: Optional[Tuple[float, float, float, float]] = None,
                     duration_minutes: int = 5) -> None:
        """
        Load AIS data from aisstream.io API or other sources.
        
        Args:
            ais_data_source: Either "aisstream" for API or file path/dict for other sources
            bbox: Bounding box (min_lat, min_lon, max_lat, max_lon) for geographic filtering
            duration_minutes: How long to collect streaming data (for aisstream API)
        """
        if ais_data_source == "aisstream":
            self._load_aisstream_data(bbox, duration_minutes)
        elif isinstance(ais_data_source, str):  # File path
            self._load_ais_from_file(ais_data_source)
        elif isinstance(ais_data_source, dict):  # Direct data
            self._parse_ais_dict(ais_data_source)
        else:
            raise ValueError(f"Unsupported AIS data source type: {type(ais_data_source)}")
    
    def _load_aisstream_data(self, bbox: Optional[Tuple[float, float, float, float]], 
                            duration_minutes: int) -> None:
        """Simple AIS data collection from aisstream.io WebSocket API."""
        
        # Basic validation
        if not self.aisstream_api_key or self.aisstream_api_key in ["your_aisstream_api_key_here", "your_api_key_here"]:
            raise ValueError("Valid AISStream API key is required")
        
        print(f"Collecting AIS data for {duration_minutes} minute(s)...")
        print(f"API Key: {self.aisstream_api_key[:8]}...")
        
        # Convert bbox format for API
        bounding_boxes = None
        if bbox:
            min_lat, min_lon, max_lat, max_lon = bbox
            bounding_boxes = [[[min_lat, min_lon], [max_lat, max_lon]]]
            print(f"Area: SW({min_lat}, {min_lon}) to NE({max_lat}, {max_lon})")
        
        # Run the collection
        asyncio.run(self._collect_ais_data(bounding_boxes, duration_minutes))
    
    async def _collect_ais_data(self, bounding_boxes: Optional[list], duration_minutes: int) -> None:
        """Simple async AIS data collection."""
        
        # Subscription message
        subscription = {
            "APIKey": self.aisstream_api_key,
            "BoundingBoxes": bounding_boxes,
            "FilterMessageTypes": ["PositionReport"]
        }
        
        try:
            print("Connecting to AISStream.io...")
            async with websockets.connect("wss://stream.aisstream.io/v0/stream") as websocket:
                
                # Send subscription
                await websocket.send(json.dumps(subscription))
                print("Connected! Receiving AIS data...")
                
                # Collect data for the specified duration
                end_time = time.time() + (duration_minutes * 60)
                message_count = 0
                
                while time.time() < end_time:
                    try:
                        # Wait for message (5 second timeout)
                        message = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                        
                        # Parse and store the message
                        data = json.loads(message)
                        self._parse_aisstream_message(data)
                        
                        message_count += 1
                        if message_count % 10 == 0:  # Progress indicator
                            print(f"Received {message_count} messages, {len(self.ais_records)} ships detected")
                    
                    except asyncio.TimeoutError:
                        continue  # Keep trying
                    except json.JSONDecodeError:
                        continue  # Skip invalid messages
                
                print(f"Collection completed: {len(self.ais_records)} ships found")
                
        except Exception as e:
            print(f"Connection error: {e}")
            raise
    
    def _parse_aisstream_message(self, data: Dict) -> None:
        """Parse AIS message and store if valid."""
        try:
            message = data.get("Message", {})
            metadata = data.get("MetaData", {})
            
            # Get essential data
            mmsi = str(metadata.get("MMSI", message.get("UserID", "")))
            latitude = message.get("Latitude")
            longitude = message.get("Longitude")
            
            # Skip if missing essential data
            if not mmsi or latitude is None or longitude is None:
                return
            
            # Parse timestamp
            timestamp_str = message.get("Timestamp", "")
            timestamp = datetime.now(timezone.utc)
            if timestamp_str:
                try:
                    timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                except:
                    pass  # Use current time if parsing fails
            
            # Create AIS record with available data
            ais_record = AISRecord(
                mmsi=mmsi,
                timestamp=timestamp,
                latitude=latitude,
                longitude=longitude,
                speed=message.get("SpeedOverGround", 0.0),
                course=message.get("CourseOverGround", 0.0),
                vessel_name=metadata.get("ShipName"),
                vessel_type=str(metadata.get("VesselType", "")) if metadata.get("VesselType") else None,
                length=None,  # Calculate if needed later
                width=None
            )
            
            self.ais_records.append(ais_record)
            
        except Exception:
            pass  # Silently skip invalid messages
    
    def _load_ais_from_file(self, file_path: str) -> None:
        """
        Load AIS data from local file (JSON or CSV format).
        """
        # TODO: Implement file loading for JSON/CSV formats
        print(f"Loading AIS data from file: {file_path}")
        pass
    
    def _parse_ais_dict(self, data: Dict) -> None:
        """
        Parse AIS data from dictionary format.
        """
        # TODO: Implement dictionary parsing
        print("Parsing AIS data from dictionary")
        pass
    
    def load_sar_detections(self, detections: List[Dict]) -> None:
        """
        Load SAR ship detections from model inference results.
        
        Programming Steps:
        1. Parse detection results from model output format
        2. Extract bounding box coordinates and convert to lat/lon
        3. Extract confidence scores and filter low-confidence detections
        4. Add timestamp information from SAR image metadata
        5. Estimate ship size from bounding box dimensions
        6. Convert to standardized SARDetection format
        7. Store in self.sar_detections list
        8. Validate detection data integrity
        
        Args:
            detections: List of detection dictionaries from SAR model
        """
        pass
    
    def calculate_spatial_distance(self, 
                                 lat1: float, lon1: float,
                                 lat2: float, lon2: float) -> float:
        """
        Calculate spatial distance between two geographic points.
        
        Programming Steps:
        1. Convert latitude/longitude to radians
        2. Apply Haversine formula for great-circle distance
        3. Account for Earth's curvature
        4. Return distance in meters
        5. Handle edge cases (same point, antipodal points)
        
        Args:
            lat1, lon1: First point coordinates
            lat2, lon2: Second point coordinates
            
        Returns:
            Distance in meters
        """
        pass
    
    def calculate_temporal_difference(self, 
                                    time1: datetime, 
                                    time2: datetime) -> float:
        """
        Calculate temporal difference between two timestamps.
        
        Programming Steps:
        1. Convert datetime objects to comparable format
        2. Calculate absolute time difference
        3. Return difference in seconds
        4. Handle timezone differences if applicable
        
        Args:
            time1, time2: Datetime objects to compare
            
        Returns:
            Time difference in seconds
        """
        pass
    
    def match_ais_to_detections(self) -> None:
        """
        Match AIS records to SAR detections using spatial-temporal correlation.
        
        Programming Steps:
        1. Create spatial index for efficient proximity searches
        2. For each SAR detection, find candidate AIS records within spatial threshold
        3. Filter candidates by temporal threshold
        4. Calculate match confidence based on:
           - Spatial proximity (closer = higher confidence)
           - Temporal proximity (recent = higher confidence)
           - Vessel size correlation (if available)
           - Movement pattern consistency
        5. Select best match for each detection (highest confidence)
        6. Handle one-to-many and many-to-one scenarios
        7. Create MatchedShip objects for all detections
        8. Store results in self.matched_ships
        """
        pass
    
    def classify_ships(self) -> Tuple[List[MatchedShip], List[MatchedShip]]:
        """
        Classify ships as AIS-equipped or non-AIS based on matching results.
        
        Programming Steps:
        1. Iterate through matched_ships list
        2. Apply classification logic:
           - Has AIS: match_confidence >= min_match_confidence AND ais_record is not None
           - No AIS: match_confidence < min_match_confidence OR ais_record is None
        3. Update has_ais flag in MatchedShip objects
        4. Separate into two lists: ais_equipped and non_ais
        5. Generate statistics for each category
        6. Handle edge cases and ambiguous matches
        
        Returns:
            Tuple of (ais_equipped_ships, non_ais_ships)
        """
        pass
    
    def generate_detection_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive report of AIS detection and matching results.
        
        Programming Steps:
        1. Calculate summary statistics:
           - Total SAR detections
           - Total AIS records processed
           - Number of successful matches
           - Number of AIS-equipped vs non-AIS ships
           - Average match confidence scores
        2. Create detailed ship inventory with:
           - Detection coordinates and confidence
           - AIS information (if matched)
           - Match quality metrics
           - Vessel characteristics
        3. Generate spatial distribution analysis
        4. Create temporal analysis of detections vs AIS activity
        5. Identify potential dark vessels (high-confidence detections without AIS)
        6. Format as structured JSON report
        7. Include metadata about processing parameters and timestamps
        
        Returns:
            Comprehensive report dictionary
        """
        pass
    
    def export_results(self, 
                      output_format: str = "json",
                      output_path: Optional[str] = None) -> str:
        """
        Export detection and matching results in specified format.
        
        Programming Steps:
        1. Generate detection report using generate_detection_report()
        2. Format output based on requested format:
           - JSON: Direct serialization of report dict
           - CSV: Flatten nested data for tabular export
           - KML/KMZ: Geographic format for mapping applications
           - GeoJSON: Standard geographic data format
        3. Handle file output if path provided, otherwise return string
        4. Include appropriate headers and metadata
        5. Validate output format and handle errors
        
        Args:
            output_format: Desired output format ("json", "csv", "kml", "geojson")
            output_path: Optional file path for output
            
        Returns:
            Formatted output string or file path
        """
        pass
    
    def visualize_matches(self, background_image: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Create visualization of AIS matches overlaid on SAR image.
        
        Programming Steps:
        1. Use background SAR image or create blank canvas
        2. Plot SAR detections as colored bounding boxes:
           - Green: AIS-equipped ships
           - Red: Non-AIS ships (potential dark vessels)
           - Yellow: Ambiguous matches
        3. Add AIS track data if available (vessel movement paths)
        4. Include legend and annotations
        5. Add confidence scores and vessel information as overlays
        6. Use different marker styles for different vessel types
        7. Ensure visualization is clear and informative
        
        Args:
            background_image: Optional SAR image for overlay
            
        Returns:
            Annotated image as numpy array
        """
        pass
    
    def get_dark_vessels(self, min_confidence: float = 0.8) -> List[MatchedShip]:
        """
        Identify potential dark vessels (ships without AIS transmissions).
        
        Programming Steps:
        1. Filter matched_ships for high-confidence SAR detections
        2. Select ships with no AIS match (has_ais = False)
        3. Apply additional filters:
           - Minimum detection confidence threshold
           - Exclude ships near ports/coastal areas (likely fishing boats)
           - Consider vessel size estimates
        4. Rank by suspicion level based on:
           - Detection confidence
           - Vessel size
           - Location (distance from shipping lanes)
           - Movement patterns if tracking data available
        5. Return sorted list of potential dark vessels
        
        Args:
            min_confidence: Minimum detection confidence for dark vessel candidates
            
        Returns:
            List of MatchedShip objects representing potential dark vessels
        """
        pass
    
    def update_tracking_data(self, tracking_results: List[Dict]) -> None:
        """
        Update matched ships with temporal tracking information.
        
        Programming Steps:
        1. Parse tracking results from DeepSort or similar tracker
        2. Match tracking IDs with existing MatchedShip objects
        3. Update movement patterns and trajectories
        4. Calculate velocity and heading information
        5. Compare with AIS movement data for validation
        6. Identify inconsistencies between SAR tracking and AIS data
        7. Update confidence scores based on movement correlation
        8. Handle new detections and lost tracks
        
        Args:
            tracking_results: List of tracking data from multi-object tracker
        """
        pass


def test_aisstream_api(api_key: Optional[str] = None, bbox: Optional[Tuple[float, float, float, float]] = None):
    """Simple test function for AIS data collection."""
    print("=== AIS Stream Test ===")
    
    # Create detector
    detector = AISDetector(aisstream_api_key=api_key)
    
    # Validate API key
    if not detector.aisstream_api_key or detector.aisstream_api_key == "your_aisstream_api_key_here":
        print("❌ No valid API key found!")
        print("Get your free API key at: https://aisstream.io/")
        print("Then run: python ais_detector.py --test-api YOUR_API_KEY")
        return
    
    # Get test area and duration
    config = detector.config.get('aisstream', {})
    test_bbox = bbox or tuple(config.get('default_bbox', [32.0, -125.0, 42.0, -115.0]))
    duration = config.get('default_duration_minutes', 1)
    
    print(f"🌍 Area: {test_bbox}")
    print(f"⏱️  Duration: {duration} minute(s)")
    
    try:
        # Collect data
        detector.load_ais_data("aisstream", bbox=test_bbox, duration_minutes=duration)
        
        # Show results
        print(f"\n✅ Found {len(detector.ais_records)} ships")
        
        if detector.ais_records:
            print("\n📡 Sample ships:")
            for i, ship in enumerate(detector.ais_records[:3]):
                print(f"  {i+1}. MMSI {ship.mmsi}: {ship.latitude:.4f}, {ship.longitude:.4f}")
                if ship.vessel_name:
                    print(f"     Name: {ship.vessel_name}")
                print(f"     Speed: {ship.speed:.1f} knots, Course: {ship.course:.0f}°")
        else:
            print("❌ No ships found - try a busier shipping area")
            
    except Exception as e:
        print(f"❌ Collection failed: {e}")
        print("💡 Check your API key and internet connection")


def main():
    """
    Example usage of AIS Detector class.
    
    This demonstrates the complete workflow for AIS detection and integration.
    """
    print("=== AIS Detector Demo ===")
    print("To test the AIS data collection, run:")
    print("python ais_detector.py --test-api YOUR_API_KEY")
    print("\nOr use the test function directly:")
    print("test_aisstream_api('your_api_key_here')")
    
    # Example initialization
    detector = AISDetector(
        spatial_threshold_meters=300.0,
        temporal_threshold_seconds=1800.0,
        min_match_confidence=0.75,
        aisstream_api_key="your_api_key_here"  # Replace with actual key
    )
    
    print(f"\nAIS Detector initialized with:")
    print(f"- Spatial threshold: {detector.spatial_threshold}m")
    print(f"- Temporal threshold: {detector.temporal_threshold}s")
    print(f"- Min match confidence: {detector.min_match_confidence}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--test-api":
        if len(sys.argv) < 3:
            print("Usage: python ais_detector.py --test-api YOUR_API_KEY")
            print("Get your free API key at: https://aisstream.io/")
            sys.exit(1)
        
        api_key = sys.argv[2]
        # Optional bounding box for testing (San Francisco Bay area by default)
        
        bbox = (32.0, -125.0, 42.0, -115.0)
        test_aisstream_api(api_key, bbox)
    else:
        main()