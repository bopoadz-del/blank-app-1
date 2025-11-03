"""
Context Detection and Classification for Reasoner AI Platform.

Automatically detects context from:
- Sensor data
- File content
- User inputs
- Historical patterns
"""
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import re
from loguru import logger


class ContextDetector:
    """
    Automatically detect and classify context from various data sources.
    """
    
    def __init__(self):
        self.climate_classifier = ClimateClassifier()
        self.material_classifier = MaterialClassifier()
        self.site_classifier = SiteConditionClassifier()
        self.project_classifier = ProjectTypeClassifier()
    
    def detect_from_text(self, text: str) -> Dict[str, Any]:
        """
        Detect context from text content.
        
        Args:
            text: Input text (from documents, descriptions, etc.)
            
        Returns:
            Detected context dictionary
        """
        text_lower = text.lower()
        
        context = {}
        
        # Detect climate
        climate = self.climate_classifier.classify(text_lower)
        if climate:
            context['climate'] = climate
        
        # Detect material
        material = self.material_classifier.classify(text_lower)
        if material:
            context['material'] = material
        
        # Detect site conditions
        site = self.site_classifier.classify(text_lower)
        if site:
            context['site_condition'] = site
        
        # Detect project type
        project = self.project_classifier.classify(text_lower)
        if project:
            context['project_type'] = project
        
        # Extract specific values
        context.update(self._extract_specific_parameters(text_lower))
        
        logger.info(f"Detected context from text: {context}")
        return context
    
    def detect_from_sensor_data(self, sensor_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detect context from sensor readings.
        
        Args:
            sensor_data: Dictionary of sensor readings
            
        Returns:
            Detected context
        """
        context = {}
        
        # Temperature-based climate detection
        if 'temperature' in sensor_data:
            temp = sensor_data['temperature']
            humidity = sensor_data.get('humidity', 50)
            
            if temp > 30:
                if humidity > 70:
                    context['climate'] = 'hot_humid'
                else:
                    context['climate'] = 'hot_arid'
            elif temp > 15:
                context['climate'] = 'temperate'
            else:
                context['climate'] = 'cold'
        
        # Pressure-based elevation detection
        if 'pressure' in sensor_data:
            pressure = sensor_data['pressure']
            # Standard pressure at sea level: 101.325 kPa
            if pressure < 95:
                context['site_condition'] = 'mountain'
            elif pressure > 101:
                context['site_condition'] = 'sea_level'
        
        # Wind-based coastal detection
        if 'wind_speed' in sensor_data:
            wind = sensor_data['wind_speed']
            if wind > 20:  # m/s
                context['site_condition'] = 'coastal'
        
        # Air quality for urban/industrial
        if 'air_quality_index' in sensor_data:
            aqi = sensor_data['air_quality_index']
            if aqi > 100:
                context['environment'] = 'industrial'
            elif aqi > 50:
                context['environment'] = 'urban'
            else:
                context['environment'] = 'rural'
        
        logger.info(f"Detected context from sensors: {context}")
        return context
    
    def detect_from_location(self, location: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detect context from geographic location.
        
        Args:
            location: Dictionary with lat, lon, city, country, etc.
            
        Returns:
            Detected context
        """
        context = {}
        
        # Latitude-based climate zones
        if 'latitude' in location:
            lat = abs(location['latitude'])
            
            if lat < 23.5:
                context['climate_zone'] = 'tropical'
            elif lat < 35:
                context['climate_zone'] = 'subtropical'
            elif lat < 60:
                context['climate_zone'] = 'temperate'
            else:
                context['climate_zone'] = 'polar'
        
        # Coastal detection (distance from coast)
        if 'distance_to_coast' in location:
            distance = location['distance_to_coast']
            if distance < 10:  # km
                context['site_condition'] = 'coastal'
            elif distance < 50:
                context['site_condition'] = 'near_coastal'
            else:
                context['site_condition'] = 'inland'
        
        # Elevation-based
        if 'elevation' in location:
            elevation = location['elevation']
            if elevation > 2000:  # meters
                context['elevation_category'] = 'high_altitude'
            elif elevation > 500:
                context['elevation_category'] = 'moderate_altitude'
            else:
                context['elevation_category'] = 'low_altitude'
        
        # Country/region-specific codes
        if 'country' in location:
            country = location['country'].upper()
            
            # Building codes
            code_mapping = {
                'US': 'ACI',
                'GB': 'BS',
                'EU': 'EN',
                'SA': 'SASO',  # Saudi Arabia
                'AE': 'UAE',
                'IN': 'IS'
            }
            
            if country in code_mapping:
                context['building_code'] = code_mapping[country]
        
        logger.info(f"Detected context from location: {context}")
        return context
    
    def detect_from_inputs(self, input_values: Dict[str, Any]) -> Dict[str, Any]:
        """
        Infer context from formula input values.
        
        Args:
            input_values: Formula input parameters
            
        Returns:
            Inferred context
        """
        context = {}
        
        # Material inference from parameters
        if 'E' in input_values:  # Elastic modulus
            E = input_values['E']
            if 190 <= E <= 210:  # GPa
                context['material'] = 'steel'
            elif 20 <= E <= 40:
                context['material'] = 'concrete'
            elif 60 <= E <= 80:
                context['material'] = 'aluminum'
        
        # Concrete grade from strength
        if 'f_c' in input_values:  # Compressive strength
            fc = input_values['f_c']
            if fc >= 50:
                context['concrete_grade'] = 'high_strength'
            elif fc >= 30:
                context['concrete_grade'] = 'normal_strength'
            else:
                context['concrete_grade'] = 'low_strength'
        
        # Temperature-based climate inference
        if 'temperature' in input_values or 'T' in input_values:
            temp = input_values.get('temperature') or input_values.get('T')
            if temp > 35:
                context['climate'] = 'hot'
            elif temp < 10:
                context['climate'] = 'cold'
        
        logger.debug(f"Inferred context from inputs: {context}")
        return context
    
    def detect_comprehensive(
        self,
        text: Optional[str] = None,
        sensor_data: Optional[Dict[str, Any]] = None,
        location: Optional[Dict[str, Any]] = None,
        input_values: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Detect context from multiple sources and merge.
        
        Args:
            text: Optional text content
            sensor_data: Optional sensor readings
            location: Optional location data
            input_values: Optional formula inputs
            
        Returns:
            Comprehensive context dictionary
        """
        context = {}
        
        if text:
            context.update(self.detect_from_text(text))
        
        if sensor_data:
            context.update(self.detect_from_sensor_data(sensor_data))
        
        if location:
            context.update(self.detect_from_location(location))
        
        if input_values:
            context.update(self.detect_from_inputs(input_values))
        
        # Add metadata
        context['detected_at'] = datetime.utcnow().isoformat()
        context['confidence'] = self._calculate_detection_confidence(context)
        
        return context
    
    def _extract_specific_parameters(self, text: str) -> Dict[str, Any]:
        """Extract specific parameters from text."""
        params = {}
        
        # Cement type
        if re.search(r'\btype\s*i\b', text):
            params['cement_type'] = 'Type_I'
        elif re.search(r'\btype\s*ii\b', text):
            params['cement_type'] = 'Type_II'
        elif re.search(r'\btype\s*v\b', text):
            params['cement_type'] = 'Type_V'
        
        # Steel grade
        steel_match = re.search(r'\b(a36|a572|a992|grade\s*(\d+))\b', text)
        if steel_match:
            params['steel_grade'] = steel_match.group(0).upper()
        
        # Exposure class
        if re.search(r'\bsevere\s+exposure\b', text):
            params['exposure_class'] = 'severe'
        elif re.search(r'\bmoderate\s+exposure\b', text):
            params['exposure_class'] = 'moderate'
        
        return params
    
    def _calculate_detection_confidence(self, context: Dict[str, Any]) -> float:
        """Calculate confidence in detected context."""
        # More context fields = higher confidence
        field_count = len([k for k in context.keys() if k != 'detected_at'])
        
        # Max confidence at 5+ fields
        confidence = min(field_count / 5.0, 1.0)
        
        return round(confidence, 2)


class ClimateClassifier:
    """Classify climate from text."""
    
    def __init__(self):
        self.patterns = {
            'hot_arid': [
                r'\b(desert|arid|hot\s+dry|low\s+humidity)\b',
                r'\b(saudi|dubai|middle\s+east)\b'
            ],
            'hot_humid': [
                r'\b(tropical|humid|monsoon|rainforest)\b',
                r'\b(singapore|mumbai|bangkok)\b'
            ],
            'temperate': [
                r'\b(temperate|moderate|mild)\b',
                r'\b(london|paris|new\s+york)\b'
            ],
            'cold': [
                r'\b(cold|arctic|freezing|sub-?zero)\b',
                r'\b(moscow|alaska|canada)\b'
            ]
        }
    
    def classify(self, text: str) -> Optional[str]:
        """Classify climate from text."""
        for climate, patterns in self.patterns.items():
            for pattern in patterns:
                if re.search(pattern, text):
                    return climate
        return None


class MaterialClassifier:
    """Classify material from text."""
    
    def __init__(self):
        self.patterns = {
            'concrete': r'\b(concrete|cement|reinforced)\b',
            'steel': r'\b(steel|iron|metal)\b',
            'aluminum': r'\b(aluminum|aluminium)\b',
            'wood': r'\b(wood|timber|lumber)\b',
            'masonry': r'\b(brick|masonry|block)\b',
            'composite': r'\b(composite|frp|fiber)\b'
        }
    
    def classify(self, text: str) -> Optional[str]:
        """Classify material from text."""
        for material, pattern in self.patterns.items():
            if re.search(pattern, text):
                return material
        return None


class SiteConditionClassifier:
    """Classify site conditions from text."""
    
    def __init__(self):
        self.patterns = {
            'coastal': r'\b(coastal|coast|beach|ocean|sea)\b',
            'mountain': r'\b(mountain|elevation|altitude|highland)\b',
            'urban': r'\b(urban|city|metropolitan)\b',
            'industrial': r'\b(industrial|factory|plant)\b',
            'marine': r'\b(marine|offshore|underwater)\b'
        }
    
    def classify(self, text: str) -> Optional[str]:
        """Classify site condition from text."""
        for condition, pattern in self.patterns.items():
            if re.search(pattern, text):
                return condition
        return None


class ProjectTypeClassifier:
    """Classify project type from text."""
    
    def __init__(self):
        self.patterns = {
            'building': r'\b(building|structure|tower|skyscraper)\b',
            'bridge': r'\b(bridge|span|viaduct)\b',
            'road': r'\b(road|highway|pavement)\b',
            'tunnel': r'\b(tunnel|underground)\b',
            'dam': r'\b(dam|reservoir|hydroelectric)\b',
            'pipeline': r'\b(pipeline|pipe|conduit)\b',
            'foundation': r'\b(foundation|pile|footing)\b'
        }
    
    def classify(self, text: str) -> Optional[str]:
        """Classify project type from text."""
        for project_type, pattern in self.patterns.items():
            if re.search(pattern, text):
                return project_type
        return None


class ContextEnricher:
    """
    Enrich detected context with additional information.
    """
    
    @staticmethod
    def enrich_with_standards(context: Dict[str, Any]) -> Dict[str, Any]:
        """Add relevant standards based on context."""
        enriched = context.copy()
        
        # Concrete standards
        if context.get('material') == 'concrete':
            enriched['applicable_standards'] = ['ACI_318', 'ASTM_C39']
        
        # Steel standards
        elif context.get('material') == 'steel':
            enriched['applicable_standards'] = ['AISC', 'ASTM_A36']
        
        # Climate-specific standards
        if context.get('climate') == 'hot_arid':
            enriched['special_considerations'] = ['thermal_expansion', 'curing_requirements']
        
        return enriched
    
    @staticmethod
    def enrich_with_constraints(context: Dict[str, Any]) -> Dict[str, Any]:
        """Add constraints based on context."""
        enriched = context.copy()
        
        constraints = []
        
        # Coastal constraints
        if context.get('site_condition') == 'coastal':
            constraints.extend(['corrosion_protection', 'salt_exposure'])
        
        # High altitude constraints
        if context.get('elevation_category') == 'high_altitude':
            constraints.extend(['low_oxygen', 'temperature_variation'])
        
        # Industrial constraints
        if context.get('environment') == 'industrial':
            constraints.extend(['chemical_exposure', 'vibration'])
        
        if constraints:
            enriched['constraints'] = constraints
        
        return enriched


# Example usage
if __name__ == "__main__":
    detector = ContextDetector()
    
    # Example 1: Text-based detection
    text = """
    This is a concrete building project located in coastal Dubai.
    The structure will be exposed to severe marine environment with
    hot and humid climate conditions. Using Type V cement for
    sulfate resistance.
    """
    
    context = detector.detect_from_text(text)
    print("Detected from text:", context)
    
    # Example 2: Sensor-based detection
    sensor_data = {
        'temperature': 38,  # Celsius
        'humidity': 75,     # %
        'pressure': 101.2,  # kPa
        'wind_speed': 25    # m/s
    }
    
    context = detector.detect_from_sensor_data(sensor_data)
    print("Detected from sensors:", context)
    
    # Example 3: Location-based detection
    location = {
        'latitude': 25.2048,
        'longitude': 55.2708,
        'city': 'Dubai',
        'country': 'AE',
        'elevation': 5,
        'distance_to_coast': 2
    }
    
    context = detector.detect_from_location(location)
    print("Detected from location:", context)
    
    # Example 4: Comprehensive detection
    context = detector.detect_comprehensive(
        text=text,
        sensor_data=sensor_data,
        location=location,
        input_values={'f_c': 50, 'E': 200}
    )
    
    # Enrich context
    enricher = ContextEnricher()
    enriched = enricher.enrich_with_standards(context)
    enriched = enricher.enrich_with_constraints(enriched)
    
    print("\nComprehensive enriched context:", enriched)
