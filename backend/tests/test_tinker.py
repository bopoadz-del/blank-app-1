"""
Test suite for Tinker ML continuous learning system.
"""
import pytest
from unittest.mock import Mock, MagicMock
from app.services.tinker import tinker_ml
from app.models.database import Formula, FormulaExecution, FormulaStatus


class TestConfidenceUpdates:
    """Test confidence score updates."""
    
    def test_confidence_increase_on_success(self):
        """Test that confidence increases on successful execution."""
        old_confidence = 0.7
        new_confidence = tinker_ml._calculate_new_confidence(
            current_confidence=old_confidence,
            total_executions=10,
            successful_executions=8,
            execution_success=True,
            error_magnitude=None
        )
        
        assert new_confidence > old_confidence
        assert 0 <= new_confidence <= 1
    
    def test_confidence_decrease_on_failure(self):
        """Test that confidence decreases on failed execution."""
        old_confidence = 0.7
        new_confidence = tinker_ml._calculate_new_confidence(
            current_confidence=old_confidence,
            total_executions=10,
            successful_executions=7,
            execution_success=False,
            error_magnitude=None
        )
        
        assert new_confidence < old_confidence
        assert 0 <= new_confidence <= 1
    
    def test_confidence_with_error_magnitude(self):
        """Test confidence calculation with error magnitude."""
        new_confidence = tinker_ml._calculate_new_confidence(
            current_confidence=0.8,
            total_executions=10,
            successful_executions=8,
            execution_success=True,
            error_magnitude=0.1  # 10% error
        )
        
        # Should increase but less than without error
        assert 0.7 < new_confidence < 0.9
    
    def test_confidence_bounds(self):
        """Test that confidence stays within [0.1, 0.99] bounds."""
        # Test upper bound
        high_confidence = tinker_ml._calculate_new_confidence(
            current_confidence=0.98,
            total_executions=100,
            successful_executions=99,
            execution_success=True
        )
        assert high_confidence <= 0.99
        
        # Test lower bound
        low_confidence = tinker_ml._calculate_new_confidence(
            current_confidence=0.15,
            total_executions=100,
            successful_executions=10,
            execution_success=False
        )
        assert low_confidence >= 0.1


class TestFormulaStatus:
    """Test formula status determination."""
    
    def test_auto_deploy_for_high_confidence_standard(self):
        """Test auto-deploy for high confidence ISO standard."""
        status = tinker_ml._determine_formula_status(
            confidence=0.96,
            total_executions=50,
            source="ISO_standard"
        )
        
        assert status == "auto_deployed"
    
    def test_approved_for_high_confidence_consultant(self):
        """Test approved status for high confidence consultant source."""
        status = tinker_ml._determine_formula_status(
            confidence=0.96,
            total_executions=50,
            source="consultant_report"
        )
        
        assert status == "approved"
    
    def test_pending_review_for_medium_confidence(self):
        """Test pending review for medium confidence."""
        status = tinker_ml._determine_formula_status(
            confidence=0.75,
            total_executions=30,
            source="consultant_report"
        )
        
        assert status == "pending_review"
    
    def test_draft_for_low_confidence_few_samples(self):
        """Test draft status for low confidence with few samples."""
        status = tinker_ml._determine_formula_status(
            confidence=0.5,
            total_executions=20,
            source="AI_discovered"
        )
        
        assert status == "draft"
    
    def test_failed_for_low_confidence_many_samples(self):
        """Test failed status for low confidence with many samples."""
        status = tinker_ml._determine_formula_status(
            confidence=0.4,
            total_executions=100,
            source="AI_discovered"
        )
        
        assert status == "failed"


class TestContextPerformance:
    """Test context-specific performance tracking."""
    
    def test_context_hash_consistency(self):
        """Test that same context produces same hash."""
        context1 = {"climate": "hot_arid", "material": "concrete"}
        context2 = {"material": "concrete", "climate": "hot_arid"}
        
        hash1 = tinker_ml._hash_context(context1)
        hash2 = tinker_ml._hash_context(context2)
        
        assert hash1 == hash2
    
    def test_context_hash_uniqueness(self):
        """Test that different contexts produce different hashes."""
        context1 = {"climate": "hot_arid"}
        context2 = {"climate": "temperate"}
        
        hash1 = tinker_ml._hash_context(context1)
        hash2 = tinker_ml._hash_context(context2)
        
        assert hash1 != hash2
    
    def test_context_similarity_identical(self):
        """Test similarity score for identical contexts."""
        context1 = {"climate": "hot_arid", "material": "concrete"}
        context2 = {"climate": "hot_arid", "material": "concrete"}
        
        similarity = tinker_ml._calculate_context_similarity(context1, context2)
        
        assert similarity == 1.0
    
    def test_context_similarity_partial(self):
        """Test similarity score for partially matching contexts."""
        context1 = {"climate": "hot_arid", "material": "concrete", "site": "coastal"}
        context2 = {"climate": "hot_arid", "material": "steel", "site": "coastal"}
        
        similarity = tinker_ml._calculate_context_similarity(context1, context2)
        
        assert 0.5 < similarity < 1.0
    
    def test_context_similarity_none(self):
        """Test similarity score for completely different contexts."""
        context1 = {"climate": "hot_arid", "material": "concrete"}
        context2 = {"region": "europe", "code": "EN1992"}
        
        similarity = tinker_ml._calculate_context_similarity(context1, context2)
        
        assert similarity == 0.0


class TestLearningInsights:
    """Test learning insights generation."""
    
    def test_confidence_reasoning_generation(self):
        """Test human-readable confidence reasoning."""
        reasoning = tinker_ml._generate_confidence_reasoning(
            old_confidence=0.7,
            new_confidence=0.75,
            execution_success=True,
            total_executions=50,
            error_magnitude=0.02
        )
        
        assert isinstance(reasoning, str)
        assert "increased" in reasoning.lower()
        assert "successful" in reasoning.lower()
    
    def test_event_id_generation(self):
        """Test unique event ID generation."""
        event_id_1 = tinker_ml._generate_event_id(1)
        event_id_2 = tinker_ml._generate_event_id(2)
        
        assert isinstance(event_id_1, str)
        assert len(event_id_1) == 32  # MD5 hash length
        assert event_id_1 != event_id_2


class TestRecommendations:
    """Test formula recommendation logic."""
    
    def test_recommendation_requires_approved_status(self):
        """Test that only approved/auto-deployed formulas are recommended."""
        # This would need database mocking in real implementation
        pass
    
    def test_recommendation_respects_min_confidence(self):
        """Test that recommendations respect minimum confidence threshold."""
        pass
    
    def test_recommendation_considers_context_match(self):
        """Test that recommendations consider context matching."""
        pass


class TestBayesianUpdate:
    """Test Bayesian confidence updating."""
    
    def test_laplace_smoothing(self):
        """Test that Laplace smoothing is applied correctly."""
        # With no executions, should start around 0.5-0.7
        confidence = tinker_ml._calculate_new_confidence(
            current_confidence=0.5,
            total_executions=0,
            successful_executions=0,
            execution_success=True
        )
        
        assert 0.4 < confidence < 0.8
    
    def test_success_rate_convergence(self):
        """Test that confidence converges to success rate over time."""
        # After many executions, confidence should approach success rate
        confidence = tinker_ml._calculate_new_confidence(
            current_confidence=0.5,
            total_executions=1000,
            successful_executions=800,  # 80% success rate
            execution_success=True
        )
        
        # Should be close to 0.8
        assert 0.75 < confidence < 0.85


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
