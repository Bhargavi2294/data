# analyze_pcb.py

import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
import json

class PCBAnalyzer:
    """Class for analyzing PCB images."""
    
    def __init__(self):
        """Initialize the PCB analyzer."""
        self.quality_model = None
        self.cert_model = None
        self.quality_classes = ['basic', 'enhanced', 'comprehensive']
        self.cert_classes = ['CE', 'RoHS', 'UL', 'FCC', 'ISO9001', 'IEC60950', 'IATF16949']
        
        # Try to load class names from files
        try:
            if os.path.exists('models/quality_check_classes.json'):
                with open('models/quality_check_classes.json', 'r') as f:
                    self.quality_classes = json.load(f)
            if os.path.exists('models/certification_classes.json'):
                with open('models/certification_classes.json', 'r') as f:
                    self.cert_classes = json.load(f)
        except Exception as e:
            print(f"Warning: Could not load class names: {e}")
    
    def load_models(self):
        """Load the trained models."""
        try:
            if os.path.exists('models/pcb_quality_check_model.h5'):
                self.quality_model = tf.keras.models.load_model('models/pcb_quality_check_model.h5')
                print("Quality check model loaded successfully.")
            else:
                print("Warning: Quality check model not found.")
                
            if os.path.exists('models/pcb_certification_model.h5'):
                self.cert_model = tf.keras.models.load_model('models/pcb_certification_model.h5')
                print("Certification model loaded successfully.")
            else:
                print("Warning: Certification model not found.")
                
            return True
        except Exception as e:
            print(f"Error loading models: {e}")
            return False
    
    def analyze_image(self, image_bytes, analysis_option=1):
        """
        Analyze a PCB image.
        
        Args:
            image_bytes: Raw image bytes
            analysis_option: 1=both, 2=quality, 3=certification
            
        Returns:
            dict: Analysis results
        """
        try:
            # Load models if not already loaded
            if self.quality_model is None or self.cert_model is None:
                self.load_models()
            
            # Convert bytes to PIL Image
            image = Image.open(io.BytesIO(image_bytes))
            
            # Resize and preprocess for model input
            img = image.resize((224, 224))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
            
            results = {}
            
            # Extract basic PCB features
            pcb_features = self.detect_pcb_features(image)
            
            # 1. Quality check analysis
            if analysis_option in [1, 2] and self.quality_model is not None:
                quality_pred = self.quality_model.predict(img_array)[0]
                quality_class_idx = np.argmax(quality_pred)
                quality_result = self.quality_classes[quality_class_idx]
                confidence = quality_pred[quality_class_idx] * 100
                
                results["quality_check_required"] = f"{quality_result} ({confidence:.1f}% confidence)"
                
                # Quality check details
                quality_details = self.get_quality_check_details(quality_result, pcb_features)
                results["quality_details"] = quality_details
            elif analysis_option in [1, 2]:
                results["quality_check_required"] = "Error: Quality check model not loaded"
            
            # 2. Certification analysis
            if analysis_option in [1, 3] and self.cert_model is not None:
                cert_pred = self.cert_model.predict(img_array)[0]
                
                # Get all certifications above threshold
                threshold = 0.5
                predicted_certs = []
                cert_confidences = {}
                
                for i, prob in enumerate(cert_pred):
                    cert_name = self.cert_classes[i]
                    confidence = prob * 100
                    cert_confidences[cert_name] = confidence
                    
                    if prob > threshold:
                        predicted_certs.append(cert_name)
                
                # Format the certification results
                if predicted_certs:
                    cert_result = "; ".join([f"{cert} ({cert_confidences[cert]:.1f}%)" for cert in predicted_certs])
                else:
                    cert_result = "No specific certifications detected"
                    
                results["certification_needed"] = cert_result
                
                # Certification details
                cert_details = self.get_certification_details(predicted_certs, pcb_features)
                results["certification_details"] = cert_details
            elif analysis_option in [1, 3]:
                results["certification_needed"] = "Error: Certification model not loaded"
            
            # Combine all details for display
            results["details"] = self.format_details(results, pcb_features, analysis_option)
            
            return results
            
        except Exception as e:
            return {
                "quality_check_required": "Error",
                "certification_needed": "Error",
                "details": f"An error occurred during image processing: {e}"
            }
    
    def detect_pcb_features(self, image):
        """
        Detect features from a PCB image.
        This is a placeholder implementation - in a real system, this would use
        more sophisticated computer vision techniques.
        
        Args:
            image: PIL Image object
            
        Returns:
            dict: Detected PCB features
        """
        # Resize for analysis
        img = image.resize((224, 224))
        img_array = np.array(img)
        
        # Simple color analysis
        mean_color = np.mean(img_array, axis=(0, 1))
        std_color = np.std(img_array, axis=(0, 1))
        
        # Simple edge detection to estimate component density
        gray = np.mean(img_array, axis=2).astype(np.uint8)
        # Simplified edge detection using std dev in local regions
        kernel_size = 5
        edge_map = np.zeros_like(gray)
        for i in range(kernel_size, gray.shape[0] - kernel_size):
            for j in range(kernel_size, gray.shape[1] - kernel_size):
                window = gray[i-kernel_size:i+kernel_size, j-kernel_size:j+kernel_size]
                edge_map[i, j] = np.std(window)
        
        edge_density = np.mean(edge_map)
        
        # Estimate PCB type and features based on color and edges
        # This is a very simplified approach - a real system would use
        # a dedicated model or more sophisticated algorithm
        
        # Estimate PCB type based on color
        pcb_type = "unknown"
        if mean_color[1] > mean_color[0] and mean_color[1] > mean_color[2]:
            # Greenish - typical FR-4
            pcb_type = "standard"
            if edge_density > 20:
                pcb_type = "multilayer"
            else:
                pcb_type = "single_sided" if edge_density < 10 else "double_sided"
        elif mean_color[2] > mean_color[0] and mean_color[2] > mean_color[1]:
            # Bluish - often high-frequency
            pcb_type = "high_frequency"
        elif mean_color[0] > mean_color[1] and mean_color[0] > mean_color[2]:
            # Reddish - sometimes high-power or specialty
            pcb_type = "high_power"
        elif np.std(mean_color) < 10:
            # Low color variation - could be flexible
            pcb_type = "flexible"
            
        # Estimate component density
        if edge_density < 10:
            component_density = "low"
        elif edge_density < 15:
            component_density = "medium"
        elif edge_density < 20:
            component_density = "high"
        else:
            component_density = "very_high"
            
        # Estimate layer count based on edge complexity
        layer_count = max(1, min(8, int(edge_density / 5)))
        
        # Check for potential issues
        issues = []
        if np.max(std_color) > 60:
            issues.append("potential color inconsistency")
        if edge_density > 25:
            issues.append("high complexity - careful inspection recommended")
            
        return {
            "pcb_type": pcb_type,
            "component_density": component_density,
            "estimated_layer_count": layer_count,
            "edge_density": edge_density,
            "issues": issues if issues else ["none detected"]
        }
    
    def get_quality_check_details(self, quality_level, features):
        """
        Get detailed quality check requirements based on quality level and PCB features.
        
        Args:
            quality_level: Predicted quality check level
            features: Detected PCB features
            
        Returns:
            list: Detailed quality check requirements
        """
        pcb_type = features["pcb_type"]
        component_density = features["component_density"]
        issues = features["issues"]
        
        # Base checks for all PCBs
        base_checks = [
            "Visual inspection for obvious defects",
            "Dimensional verification",
            "Solder joint inspection"
        ]
        
        # Additional checks based on quality level
        additional_checks = []
        
        if quality_level == "basic":
            additional_checks = [
                "Basic continuity testing",
                "Simple functional testing"
            ]
        elif quality_level == "enhanced":
            additional_checks = [
                "Automated Optical Inspection (AOI)",
                "Complete continuity and isolation testing",
                "Functional testing with basic parameters"
            ]
        elif quality_level == "comprehensive":
            additional_checks = [
                "Automated Optical Inspection (AOI)",
                "Automated X-ray Inspection (AXI)",
                "In-Circuit Testing (ICT)",
                "Flying Probe Testing",
                "Functional testing with extended parameters",
                "Thermal stress testing"
            ]
            
        # PCB type specific checks
        type_specific_checks = []
        
        if pcb_type == "multilayer":
            type_specific_checks.append("Layer-to-layer registration verification")
            type_specific_checks.append("Buried/blind via inspection")
        elif pcb_type == "flexible" or pcb_type == "rigid_flex":
            type_specific_checks.append("Flexibility and bend testing")
            type_specific_checks.append("Delamination inspection")
        elif pcb_type == "high_frequency":
            type_specific_checks.append("Impedance testing")
            type_specific_checks.append("Signal integrity verification")
        elif pcb_type == "high_power":
            type_specific_checks.append("Copper thickness verification")
            type_specific_checks.append("Thermal performance testing")
            
        # Add issue-specific checks
        issue_specific_checks = []
        for issue in issues:
            if issue != "none detected":
                issue_specific_checks.append(f"Detailed inspection for {issue}")
                
        # Combine all checks
        all_checks = base_checks + additional_checks + type_specific_checks + issue_specific_checks
        
        return all_checks
    
    def get_certification_details(self, certifications, features):
        """
        Get detailed certification requirements.
        
        Args:
            certifications: List of predicted certifications
            features: Detected PCB features
            
        Returns:
            dict: Certification details and requirements
        """
        cert_details = {}
        
        for cert in certifications:
            if cert == "CE":
                cert_details["CE"] = {
                    "description": "European Conformity - Required for products sold in EU",
                    "requirements": [
                        "EMC Directive compliance",
                        "RoHS compliance",
                        "Safety testing",
                        "Technical documentation"
                    ]
                }
            elif cert == "RoHS":
                cert_details["RoHS"] = {
                    "description": "Restriction of Hazardous Substances - Environmental standard",
                    "requirements": [
                        "No lead, mercury, cadmium, hexavalent chromium, PBBs, PBDEs",
                        "Test reports for materials",
                        "Declaration of Conformity"
                    ]
                }
            elif cert == "UL":
                cert_details["UL"] = {
                    "description": "Underwriters Laboratories - Safety standard",
                    "requirements": [
                        "Safety testing",
                        "Flammability testing",
                        "Regular factory audits",
                        "UL mark application"
                    ]
                }
            elif cert == "FCC":
                cert_details["FCC"] = {
                    "description": "Federal Communications Commission - US EMC standard",
                    "requirements": [
                        "EMI/EMC testing",
                        "Radiated and conducted emissions testing",
                        "Technical documentation",
                        "FCC Declaration of Conformity or Certification"
                    ]
                }
            elif cert == "ISO9001":
                cert_details["ISO9001"] = {
                    "description": "Quality Management System standard",
                    "requirements": [
                        "Documented quality procedures",
                        "Regular audits",
                        "Continual improvement processes",
                        "Management reviews"
                    ]
                }
            elif cert == "IEC60950":
                cert_details["IEC60950"] = {
                    "description": "Information Technology Equipment Safety",
                    "requirements": [
                        "Electrical safety testing",
                        "Thermal testing",
                        "Mechanical strength testing",
                        "Fire enclosure requirements"
                    ]
                }
            elif cert == "IATF16949":
                cert_details["IATF16949"] = {
                    "description": "Automotive Quality Management System",
                    "requirements": [
                        "Automotive-specific quality processes",
                        "Production part approval process (PPAP)",
                        "Failure mode and effects analysis (FMEA)",
                        "Statistical process control"
                    ]
                }
                
        return cert_details
    
    def format_details(self, results, features, analysis_option):
        """
        Format all details for display.
        
        Args:
            results: Analysis results
            features: Detected PCB features
            analysis_option: Analysis option selected
            
        Returns:
            str: Formatted details
        """
        details = []
        
        # Add PCB features
        details.append(f"PCB Type: {features['pcb_type'].upper()}")
        details.append(f"Component Density: {features['component_density'].capitalize()}")
        details.append(f"Estimated Layer Count: {features['estimated_layer_count']}")
        
        # Add detected issues
        if features['issues'] and features['issues'][0] != "none detected":
            details.append("Detected Issues: " + ", ".join(features['issues']))
        else:
            details.append("Detected Issues: None")
            
        details.append("\n")
            
        # Add quality check details
        if analysis_option in [1, 2] and "quality_details" in results:
            details.append("RECOMMENDED QUALITY CHECKS:")
            for i, check in enumerate(results["quality_details"], 1):
                details.append(f"{i}. {check}")
            details.append("\n")
            
        # Add certification details
        if analysis_option in [1, 3] and "certification_details" in results:
            if results["certification_details"]:
                details.append("CERTIFICATION REQUIREMENTS:")
                for cert, info in results["certification_details"].items():
                    details.append(f"• {cert}: {info['description']}")
                    details.append("  Requirements:")
                    for req in info['requirements']:
                        details.append(f"  - {req}")
                    details.append("")
            else:
                details.append("CERTIFICATION REQUIREMENTS: None specifically detected")
                
        return "\n".join(details)


# Function to use in Streamlit app
def analyze_pcb_image(image_bytes, analysis_option):
    """
    Analyze a PCB image using trained ML models.
    
    Args:
        image_bytes (bytes): The raw bytes of the uploaded image.
        analysis_option (int): The selected analysis option (1, 2, or 3).
        
    Returns:
        dict: A dictionary containing the analysis results.
    """
    analyzer = PCBAnalyzer()
    return analyzer.analyze_image(image_bytes, analysis_option)


# Example usage
if __name__ == "__main__":
    print("This module provides the PCB image analysis functionality.")
    print("It should be imported and used in the Streamlit app.")
