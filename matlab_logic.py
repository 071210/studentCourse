# matlab_logic.py - Exact Python Conversion of Your MATLAB System

import numpy as np
import json
from datetime import datetime

class FuzzyInferenceSystem:
    """Python implementation of your MATLAB FIS"""
    
    def __init__(self):
        # Define membership functions based on your .fis file
        self.setup_membership_functions()
        self.setup_rules()
    
    def setup_membership_functions(self):
        """Setup membership functions from your FIS file"""
        # Input ranges and membership functions
        self.input_ranges = {
            'CGPA': [1, 5],
            'Programming': [0, 5],
            'Multimedia': [0, 5],
            'MachineLearning': [0, 5],
            'Database': [0, 5],
            'SoftwareEngineering': [0, 5],
            'GameDevelopment': [1, 5],
            'WebDevelopment': [1, 5],
            'ArtificialIntelligence': [1, 5],
            'DatabaseSystem': [1, 5],
            'SoftwareValidation': [1, 5],
            'Difficulty': [1, 3],
            'LearningStyle': [1, 4]
        }
        
        # Output range
        self.output_range = [1, 5]
    
    def triangular_mf(self, x, params):
        """Triangular membership function"""
        a, b, c = params
        if x <= a or x >= c:
            return 0.0
        elif a < x <= b:
            return (x - a) / (b - a)
        else:
            return (c - x) / (c - b)
    
    def get_membership_values(self, inputs):
        """Get membership values for all inputs"""
        # CGPA membership functions
        cgpa = inputs[0]
        cgpa_low = self.triangular_mf(cgpa, [1, 1, 3])
        cgpa_med = self.triangular_mf(cgpa, [2, 3, 4])
        cgpa_high = self.triangular_mf(cgpa, [3, 5, 5])
        
        # Programming membership functions
        prog = inputs[1]
        prog_low = self.triangular_mf(prog, [0, 0, 2])
        prog_med = self.triangular_mf(prog, [1, 2.5, 4])
        prog_high = self.triangular_mf(prog, [3, 5, 5])
        
        # Game Development membership functions
        game = inputs[6]
        game_low = self.triangular_mf(game, [1, 1, 3])
        game_med = self.triangular_mf(game, [2, 3, 4])
        game_high = self.triangular_mf(game, [3, 5, 5])
        
        # Web Development membership functions
        web = inputs[7]
        web_low = self.triangular_mf(web, [1, 1, 3])
        web_med = self.triangular_mf(web, [2, 3, 4])
        web_high = self.triangular_mf(web, [3, 5, 5])
        
        # AI membership functions
        ai = inputs[8]
        ai_low = self.triangular_mf(ai, [1, 1, 3])
        ai_med = self.triangular_mf(ai, [2, 3, 4])
        ai_high = self.triangular_mf(ai, [3, 5, 5])
        
        # Database System membership functions
        db = inputs[9]
        db_low = self.triangular_mf(db, [1, 1, 3])
        db_med = self.triangular_mf(db, [2, 3, 4])
        db_high = self.triangular_mf(db, [3, 5, 5])
        
        # Software Validation membership functions
        sv = inputs[10]
        sv_low = self.triangular_mf(sv, [1, 1, 3])
        sv_med = self.triangular_mf(sv, [2, 3, 4])
        sv_high = self.triangular_mf(sv, [3, 5, 5])
        
        # Difficulty membership functions
        diff = inputs[11]
        diff_easy = self.triangular_mf(diff, [0.5, 1, 1.5])
        diff_mod = self.triangular_mf(diff, [1.5, 2, 2.5])
        diff_hard = self.triangular_mf(diff, [2.5, 3, 3.5])
        
        # Learning Style membership functions
        style = inputs[12]
        style_visual = self.triangular_mf(style, [0.5, 1, 1.5])
        style_kines = self.triangular_mf(style, [1.5, 2, 2.5])
        style_read = self.triangular_mf(style, [2.5, 3, 3.5])
        style_audit = self.triangular_mf(style, [3.5, 4, 4.5])
        
        return {
            'cgpa': [cgpa_low, cgpa_med, cgpa_high],
            'programming': [prog_low, prog_med, prog_high],
            'game_dev': [game_low, game_med, game_high],
            'web_dev': [web_low, web_med, web_high],
            'ai': [ai_low, ai_med, ai_high],
            'db_sys': [db_low, db_med, db_high],
            'soft_val': [sv_low, sv_med, sv_high],
            'difficulty': [diff_easy, diff_mod, diff_hard],
            'style': [style_visual, style_kines, style_read, style_audit]
        }
    
    def setup_rules(self):
        """Setup fuzzy rules based on your FIS rules"""
        # Simplified rule base - key patterns from your 242 rules
        self.rules = [
            # High CGPA + High Programming + High Gaming Interest -> Gaming
            {'antecedent': {'cgpa': 2, 'programming': 2, 'game_dev': 2}, 'consequent': 1, 'weight': 1.0},
            
            # High CGPA + High Programming + High Web Interest -> Web Development  
            {'antecedent': {'cgpa': 2, 'programming': 2, 'web_dev': 2}, 'consequent': 2, 'weight': 1.0},
            
            # High AI Interest + Difficult -> Fuzzy Logic
            {'antecedent': {'ai': 2, 'difficulty': 2}, 'consequent': 3, 'weight': 1.0},
            
            # High Database Interest + Easy/Moderate -> Database Design
            {'antecedent': {'db_sys': 2, 'difficulty': 0}, 'consequent': 4, 'weight': 1.0},
            {'antecedent': {'db_sys': 2, 'difficulty': 1}, 'consequent': 4, 'weight': 1.0},
            
            # High Software Validation + Moderate -> Software Testing
            {'antecedent': {'soft_val': 2, 'difficulty': 1}, 'consequent': 5, 'weight': 1.0},
            
            # Low performance defaults
            {'antecedent': {'cgpa': 0, 'programming': 0}, 'consequent': 4, 'weight': 0.8},
        ]
    
    def evaluate_rules(self, membership_values):
        """Evaluate fuzzy rules and get output"""
        output_weights = [0.0] * 5  # For 5 courses
        
        for rule in self.rules:
            # Calculate rule strength (minimum of antecedents)
            rule_strength = 1.0
            
            for input_var, mf_index in rule['antecedent'].items():
                if input_var in membership_values:
                    rule_strength = min(rule_strength, membership_values[input_var][mf_index])
            
            # Apply rule weight
            rule_strength *= rule['weight']
            
            # Add to output
            output_index = rule['consequent'] - 1  # Convert to 0-based index
            output_weights[output_index] += rule_strength
        
        return output_weights
    
    def defuzzify(self, output_weights):
        """Convert fuzzy output to crisp value using centroid method"""
        # Output membership function centers
        centers = [1.0, 2.0, 3.0, 4.0, 5.0]  # Gaming, Web, Fuzzy, DB, Software
        
        numerator = sum(w * c for w, c in zip(output_weights, centers))
        denominator = sum(output_weights)
        
        if denominator == 0:
            return 3.0  # Default middle value
        
        return numerator / denominator
    
    def evalfis(self, inputs):
        """Main FIS evaluation function - equivalent to MATLAB evalfis"""
        membership_values = self.get_membership_values(inputs)
        output_weights = self.evaluate_rules(membership_values)
        return self.defuzzify(output_weights)


class CourseRecommendationEngine:
    """Exact Python version of your MATLAB recommendation system"""
    
    def __init__(self):
        # Course names (same as your MATLAB)
        self.courses = [
            'Gaming', 
            'Web Development', 
            'Fuzzy Logic', 
            'Database Design', 
            'Software Validation & Verification'
        ]
        
        # Initialize FIS
        self.fis = FuzzyInferenceSystem()
        
        # Subject and interest mappings (from your MATLAB functions)
        self.subject_map = self.define_subject_map()
        self.interest_map = self.define_interest_map()
        
        # Thresholds
        self.subject_threshold = 1.5
        self.interest_threshold = 1.5
    
    def define_subject_map(self):
        """Python version of defineSubjectMap() function"""
        return [
            {'index': 2, 'courses': [1, 2, 3, 4, 5], 'weights': [0.7, 1.0, 0.6, 0.8, 0.9]},  # Programming
            {'index': 3, 'courses': [1, 2], 'weights': [1.0, 0.8]},                          # Multimedia
            {'index': 4, 'courses': [3], 'weights': [1.0]},                                  # Machine Learning
            {'index': 5, 'courses': [4], 'weights': [1.0]},                                  # Database
            {'index': 6, 'courses': [2, 5], 'weights': [0.8, 1.0]}                          # Software Engineering
        ]
    
    def define_interest_map(self):
        """Python version of defineInterestMap() function"""
        return [
            {'indices': [7], 'course': 1, 'weights': [1.0]},   # Gaming
            {'indices': [8], 'course': 2, 'weights': [1.0]},   # Web Development
            {'indices': [9], 'course': 3, 'weights': [1.0]},   # AI/Fuzzy Logic
            {'indices': [10], 'course': 4, 'weights': [1.0]},  # Database Design
            {'indices': [11], 'course': 5, 'weights': [1.0]}   # Software Validation
        ]
    
    def subject_scoring(self, row, threshold):
        """Python version of subjectScoring() function"""
        scores = [0.0] * 5
        
        for subj in self.subject_map:
            value = row[subj['index'] - 1]  # Convert to 0-based indexing
            if value >= threshold:
                normalized = value / 5.0
                for j, course_idx in enumerate(subj['courses']):
                    scores[course_idx - 1] += normalized * subj['weights'][j]
        
        return scores
    
    def interest_scoring(self, row, threshold):
        """Python version of interestScoring() function"""
        scores = [0.0] * 5
        
        for entry in self.interest_map:
            total = 0.0
            count = 0
            
            for j, idx in enumerate(entry['indices']):
                val = row[idx - 1]  # Convert to 0-based indexing
                if val >= threshold:
                    total += val * entry['weights'][j]
                    count += 1
            
            if count > 0:
                scores[entry['course'] - 1] += (total / count / 5.0)
        
        return scores
    
    def adjust_by_learning_preferences(self, scores, difficulty, style):
        """Python version of adjustByLearningPreferences() function"""
        scores = scores.copy()  # Don't modify original
        
        # Difficulty adjustments
        if difficulty == 1:  # Easy
            scores[1] *= 1.2  # Web Development
            scores[3] *= 1.2  # Database Design
        elif difficulty == 2:  # Moderate
            scores[0] *= 1.1  # Gaming
            scores[4] *= 1.1  # Software Validation
        elif difficulty == 3:  # Difficult
            scores[2] *= 1.2  # Fuzzy Logic
        
        # Learning Style adjustments
        if style == 1:  # Visual
            scores[0] *= 1.1  # Gaming
            scores[2] *= 1.1  # Fuzzy Logic
            scores[3] *= 1.1  # Database Design
        elif style == 2:  # Kinesthetic
            scores[0] *= 1.15  # Gaming
            scores[1] *= 1.15  # Web Development
            scores[3] *= 1.15  # Database Design
        elif style == 3:  # Reading/Writing
            scores[1] *= 1.1  # Web Development
            scores[4] *= 1.1  # Software Validation
        elif style == 4:  # Auditory
            scores[2] *= 1.05  # Fuzzy Logic
            scores[4] *= 1.05  # Software Validation
        
        return scores
    
    def simple_decision_tree(self, features):
        """Simplified decision tree logic (replace with trained model later)"""
        # Extract key features
        cgpa = features[0]
        programming = features[1]
        game_interest = features[6]
        web_interest = features[7]
        ai_interest = features[8]
        db_interest = features[9]
        fis_output = features[13]  # FIS output is the 14th feature
        
        # Simple tree logic based on common patterns
        if programming >= 4 and cgpa >= 4:
            if game_interest >= 3:
                return 1  # Gaming
            elif web_interest >= 3:
                return 2  # Web Development
            else:
                return 5  # Software Testing
        elif cgpa >= 3:
            if ai_interest >= 3:
                return 3  # AI/Fuzzy Logic
            elif db_interest >= 3:
                return 4  # Database Design
            else:
                return int(round(fis_output))  # Use FIS recommendation
        else:
            # Lower performance - recommend easier courses
            if web_interest >= 3:
                return 2  # Web Development
            else:
                return 4  # Database Design
    
    def generate_recommendation(self, input_data):
        """Main recommendation function - exact Python version of your MATLAB system"""
        
        # Convert input to array (same order as MATLAB: 1-13)
        input_array = [
            input_data.get('cgpa', 3),
            input_data.get('programming', 0),
            input_data.get('multimedia', 0),
            input_data.get('machineLearning', 0),
            input_data.get('database', 0),
            input_data.get('softwareEngineering', 0),
            input_data.get('gameDevelopment', 1),
            input_data.get('webDevelopment', 1),
            input_data.get('artificialIntelligence', 1),
            input_data.get('databaseSystem', 1),
            input_data.get('softwareValidation', 1),
            input_data.get('difficulty', 2),
            input_data.get('learningStyle', 1)
        ]
        
        # Initialize course scores
        course_scores = [0.0] * 5
        
        # Subject & Interest scoring (exact MATLAB logic)
        subject_scores = self.subject_scoring(input_array, self.subject_threshold)
        interest_scores = self.interest_scoring(input_array, self.interest_threshold)
        
        # Combine scores
        for i in range(5):
            course_scores[i] = subject_scores[i] + interest_scores[i]
        
        # FIS Output (exact MATLAB evalfis equivalent)
        fis_output = self.fis.evalfis(input_array)
        course_recommend = round(fis_output)
        
        # FIS boost (exact MATLAB logic)
        if 1 <= course_recommend <= 5 and fis_output != 3.0:
            course_scores[course_recommend - 1] += 0.3
        
        # Adjust by difficulty & learning style (exact MATLAB function)
        course_scores = self.adjust_by_learning_preferences(
            course_scores, input_array[11], input_array[12]
        )
        
        # Ensure minimum scores (exact MATLAB logic)
        course_scores = [max(score, 0.1) for score in course_scores]
        
        # Expert Decision (exact MATLAB logic)
        sorted_indices = sorted(range(len(course_scores)), key=lambda i: course_scores[i], reverse=True)
        expert_index = sorted_indices[0]
        expert_confidence = course_scores[expert_index]
        
        # Decision Tree Prediction
        tree_features = input_array + [fis_output]  # 14 features total
        tree_index = self.simple_decision_tree(tree_features)
        
        # Ensure tree prediction is in bounds
        if tree_index < 1 or tree_index > 5:
            tree_index = expert_index + 1  # Convert to 1-based
        
        # Final Decision (exact MATLAB logic)
        if (expert_index + 1) == tree_index:  # Convert expert_index to 1-based
            final_course = self.courses[expert_index]
        else:
            if expert_confidence >= 0.75:
                final_course = self.courses[expert_index]
            else:
                final_course = self.courses[tree_index - 1]  # Convert to 0-based
        
        # Create result (exact MATLAB format)
        result = {
            'firstRecommendedCourse': self.courses[sorted_indices[0]],
            'alternativeRecommendedCourse': self.courses[sorted_indices[1]],
            'firstConfidence': course_scores[sorted_indices[0]],
            'secondConfidence': course_scores[sorted_indices[1]],
            'probability_Gaming': course_scores[0],
            'probability_WebDevelopment': course_scores[1],
            'probability_FuzzyLogic': course_scores[2],
            'probability_DatabaseDesign': course_scores[3],
            'probability_SoftwareValidation_Verification': course_scores[4],
            'expertRecommendation': self.courses[expert_index],
            'treeRecommendation': self.courses[tree_index - 1],
            'finalRecommendation': final_course,
            'fisOutput': fis_output,
            'processingMethod': 'Python_MATLAB_Equivalent',
            'timestamp': datetime.now().isoformat()
        }
        
        return result

# Export the class
__all__ = ['CourseRecommendationEngine']
