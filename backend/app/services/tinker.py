"""
Tinker ML - Continuous Learning System for The Reasoner Platform.
Tracks formula performance and updates confidence scores dynamically.
"""
import hashlib
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np
from sqlalchemy.orm import Session
from sqlalchemy import func, and_
from loguru import logger

from app.models.database import (
    Formula, FormulaExecution, ContextPerformance, 
    LearningEvent, ExecutionStatus
)
from app.core.config import settings


class TinkerML:
    """
    Continuous learning system that:
    1. Tracks formula execution success/failure
    2. Updates confidence scores based on performance
    3. Learns context-specific performance patterns
    4. Determines auto-deploy vs human review requirements
    """
    
    def __init__(self):
        self.auto_deploy_threshold = settings.AUTO_DEPLOY_CONFIDENCE_THRESHOLD
        self.human_review_threshold = settings.HUMAN_REVIEW_CONFIDENCE_THRESHOLD
        self.min_samples = settings.MIN_SAMPLES_FOR_CONFIDENCE
        self.decay_rate = settings.CONFIDENCE_DECAY_RATE
        self.growth_rate = settings.CONFIDENCE_GROWTH_RATE
    
    async def update_confidence_from_execution(
        self,
        db: Session,
        formula_id: int,
        execution_success: bool,
        context: Optional[Dict[str, Any]] = None,
        error_magnitude: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Update formula confidence based on execution result.
        
        Args:
            db: Database session
            formula_id: Formula ID
            execution_success: Whether execution succeeded
            context: Execution context (climate, materials, etc.)
            error_magnitude: Relative error if validation data available
            
        Returns:
            Dictionary with old confidence, new confidence, and reasoning
        """
        formula = db.query(Formula).filter(Formula.id == formula_id).first()
        if not formula:
            raise ValueError(f"Formula {formula_id} not found")
        
        old_confidence = formula.confidence_score
        
        # Update execution counts
        formula.total_executions += 1
        if execution_success:
            formula.successful_executions += 1
        else:
            formula.failed_executions += 1
        
        # Calculate new confidence using Bayesian update
        new_confidence = self._calculate_new_confidence(
            current_confidence=old_confidence,
            total_executions=formula.total_executions,
            successful_executions=formula.successful_executions,
            execution_success=execution_success,
            error_magnitude=error_magnitude
        )
        
        formula.confidence_score = new_confidence
        
        # Update context-specific performance if context provided
        context_update = None
        if context:
            context_update = await self._update_context_performance(
                db, formula_id, context, execution_success, error_magnitude
            )
        
        # Determine if status should change based on confidence
        old_status = formula.status
        new_status = self._determine_formula_status(
            new_confidence,
            formula.total_executions,
            formula.source
        )
        
        if new_status != old_status:
            formula.status = new_status
        
        # Create learning event
        learning_event = LearningEvent(
            event_id=self._generate_event_id(formula_id),
            event_type="confidence_update",
            formula_id=formula_id,
            old_confidence=old_confidence,
            new_confidence=new_confidence,
            reason=f"Execution {'succeeded' if execution_success else 'failed'}",
            evidence_data={
                "execution_success": execution_success,
                "error_magnitude": error_magnitude,
                "context": context,
                "total_executions": formula.total_executions,
                "success_rate": formula.successful_executions / formula.total_executions
            },
            triggered_by="system"
        )
        db.add(learning_event)
        db.commit()
        
        logger.info(
            f"Updated confidence for formula {formula.formula_id}: "
            f"{old_confidence:.3f} -> {new_confidence:.3f} "
            f"(status: {old_status} -> {new_status})"
        )
        
        return {
            "formula_id": formula.formula_id,
            "old_confidence": old_confidence,
            "new_confidence": new_confidence,
            "confidence_change": new_confidence - old_confidence,
            "old_status": old_status,
            "new_status": new_status,
            "status_changed": new_status != old_status,
            "total_executions": formula.total_executions,
            "success_rate": formula.successful_executions / formula.total_executions,
            "context_performance": context_update,
            "reasoning": self._generate_confidence_reasoning(
                old_confidence, new_confidence, execution_success, 
                formula.total_executions, error_magnitude
            )
        }
    
    def _calculate_new_confidence(
        self,
        current_confidence: float,
        total_executions: int,
        successful_executions: int,
        execution_success: bool,
        error_magnitude: Optional[float] = None
    ) -> float:
        """
        Calculate new confidence score using Bayesian updating.
        
        Confidence formula:
        - Base: Success rate with Laplace smoothing
        - Adjustment: Recent execution impact
        - Error penalty: If error magnitude provided
        """
        # Calculate success rate with Laplace smoothing
        alpha = 2.0  # Prior successes
        beta = 1.0   # Prior failures
        
        success_rate = (successful_executions + alpha) / (total_executions + alpha + beta)
        
        # Weight recent execution more heavily for early samples
        recency_weight = min(1.0, self.min_samples / max(total_executions, 1))
        
        # Calculate confidence change
        if execution_success:
            confidence_delta = self.growth_rate * recency_weight
            if error_magnitude is not None and error_magnitude > 0:
                # Reduce growth if there's error even on "success"
                confidence_delta *= (1.0 - min(error_magnitude, 0.5))
        else:
            confidence_delta = -self.decay_rate * recency_weight
        
        # New confidence is weighted average of success rate and adjusted current
        new_confidence = (
            0.85 * success_rate
            + 0.15 * (current_confidence + confidence_delta)
        )
        
        # Confidence bounds: [0.1, 0.99]
        new_confidence = max(0.1, min(0.99, new_confidence))
        
        return new_confidence
    
    async def _update_context_performance(
        self,
        db: Session,
        formula_id: int,
        context: Dict[str, Any],
        execution_success: bool,
        error_magnitude: Optional[float] = None
    ) -> Dict[str, Any]:
        """Update performance tracking for specific context."""
        context_hash = self._hash_context(context)
        
        # Find or create context performance record
        ctx_perf = db.query(ContextPerformance).filter(
            and_(
                ContextPerformance.formula_id == formula_id,
                ContextPerformance.context_hash == context_hash
            )
        ).first()
        
        if not ctx_perf:
            ctx_perf = ContextPerformance(
                formula_id=formula_id,
                context_hash=context_hash,
                context_data=context,
                total_executions=0,
                successful_executions=0,
                average_error=0.0,
                confidence_in_context=0.5
            )
            db.add(ctx_perf)
        
        # Update counts
        ctx_perf.total_executions += 1
        if execution_success:
            ctx_perf.successful_executions += 1
        
        # Update average error
        if error_magnitude is not None:
            if ctx_perf.average_error == 0.0:
                ctx_perf.average_error = error_magnitude
            else:
                # Exponential moving average
                alpha = 0.3
                ctx_perf.average_error = (
                    alpha * error_magnitude + 
                    (1 - alpha) * ctx_perf.average_error
                )
        
        # Calculate context-specific confidence
        ctx_success_rate = ctx_perf.successful_executions / ctx_perf.total_executions
        
        # Adjust for error magnitude
        error_penalty = 0.0
        if ctx_perf.average_error > 0:
            error_penalty = min(ctx_perf.average_error, 0.3)
        
        ctx_perf.confidence_in_context = max(
            0.1,
            min(0.99, ctx_success_rate - error_penalty)
        )
        
        ctx_perf.last_execution_date = datetime.utcnow()
        db.commit()
        
        return {
            "context_hash": context_hash,
            "total_executions": ctx_perf.total_executions,
            "success_rate": ctx_success_rate,
            "average_error": ctx_perf.average_error,
            "confidence_in_context": ctx_perf.confidence_in_context
        }
    
    def _determine_formula_status(
        self,
        confidence: float,
        total_executions: int,
        source: Optional[str]
    ) -> str:
        """
        Determine formula status based on confidence and execution history.
        
        Credibility-based autonomy framework:
        - High confidence + ISO/standard source -> AUTO_DEPLOYED
        - High confidence + consultant/AI source -> APPROVED (needs review)
        - Medium confidence -> PENDING_REVIEW
        - Low confidence -> DRAFT or FAILED
        """
        if total_executions < self.min_samples:
            return "pending_review"
        
        # High confidence: can be auto-deployed if from trusted source
        if confidence >= self.auto_deploy_threshold:
            if source and any(trusted in source.lower() for trusted in ['iso', 'astm', 'aci', 'standard']):
                return "auto_deployed"
            else:
                return "approved"
        
        # Medium confidence: needs human review
        elif confidence >= self.human_review_threshold:
            return "pending_review"
        
        # Low confidence: keep as draft or mark failed
        else:
            if total_executions > 50:
                return "failed"
            else:
                return "draft"
    
    def recommend_formulas(
        self,
        db: Session,
        domain: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        min_confidence: float = 0.5,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Recommend formulas based on domain, context, and confidence.
        
        Uses context-aware scoring to recommend formulas that perform
        well in similar contexts.
        """
        query = db.query(Formula).filter(
            Formula.confidence_score >= min_confidence,
            Formula.status.in_(['approved', 'auto_deployed'])
        )
        
        if domain:
            query = query.filter(Formula.domain == domain)
        
        formulas = query.order_by(Formula.confidence_score.desc()).limit(limit * 2).all()
        
        recommendations = []
        for formula in formulas:
            # Calculate match score based on context
            match_score = self._calculate_context_match_score(
                db, formula.id, context
            )
            
            # Combined score: confidence * match_score
            combined_score = formula.confidence_score * match_score
            
            recommendations.append({
                "formula_id": formula.formula_id,
                "name": formula.name,
                "description": formula.description,
                "domain": formula.domain,
                "confidence_score": formula.confidence_score,
                "match_score": match_score,
                "combined_score": combined_score,
                "total_executions": formula.total_executions,
                "success_rate": (
                    formula.successful_executions / formula.total_executions
                    if formula.total_executions > 0 else 0.0
                )
            })
        
        # Sort by combined score
        recommendations.sort(key=lambda x: x['combined_score'], reverse=True)
        
        return recommendations[:limit]
    
    def _calculate_context_match_score(
        self,
        db: Session,
        formula_id: int,
        context: Optional[Dict[str, Any]]
    ) -> float:
        """
        Calculate how well a formula matches the given context.
        Returns score between 0.0 (no match) and 1.0 (perfect match).
        """
        if not context:
            return 0.8  # Default score when no context provided
        
        context_hash = self._hash_context(context)
        
        # Check if we have performance data for this exact context
        ctx_perf = db.query(ContextPerformance).filter(
            and_(
                ContextPerformance.formula_id == formula_id,
                ContextPerformance.context_hash == context_hash
            )
        ).first()
        
        if ctx_perf and ctx_perf.total_executions >= 3:
            # We have data for this exact context
            return ctx_perf.confidence_in_context
        
        # Check for similar contexts (partial matches)
        all_ctx_perfs = db.query(ContextPerformance).filter(
            ContextPerformance.formula_id == formula_id,
            ContextPerformance.total_executions >= 3
        ).all()
        
        if not all_ctx_perfs:
            return 0.7  # No context data, use moderate default
        
        # Calculate similarity scores with existing contexts
        similarities = []
        for ctx_perf in all_ctx_perfs:
            similarity = self._calculate_context_similarity(
                context, ctx_perf.context_data
            )
            weighted_confidence = similarity * ctx_perf.confidence_in_context
            similarities.append(weighted_confidence)
        
        # Return weighted average
        return np.mean(similarities) if similarities else 0.7
    
    def _calculate_context_similarity(
        self,
        context1: Dict[str, Any],
        context2: Dict[str, Any]
    ) -> float:
        """
        Calculate similarity between two contexts.
        Returns value between 0.0 (no match) and 1.0 (identical).
        """
        all_keys = set(context1.keys()) | set(context2.keys())
        if not all_keys:
            return 1.0
        
        matches = 0
        for key in all_keys:
            val1 = context1.get(key)
            val2 = context2.get(key)
            
            if val1 == val2:
                matches += 1
            elif val1 is not None and val2 is not None:
                # Partial match for string similarity
                if isinstance(val1, str) and isinstance(val2, str):
                    if val1.lower() in val2.lower() or val2.lower() in val1.lower():
                        matches += 0.5
        
        return matches / len(all_keys)
    
    def _hash_context(self, context: Dict[str, Any]) -> str:
        """Generate hash for context dictionary."""
        # Sort keys for consistent hashing
        context_str = str(sorted(context.items()))
        return hashlib.md5(context_str.encode()).hexdigest()
    
    def _generate_event_id(self, formula_id: int) -> str:
        """Generate unique event ID."""
        timestamp = datetime.utcnow().isoformat()
        return hashlib.md5(f"{formula_id}:{timestamp}".encode()).hexdigest()
    
    def _generate_confidence_reasoning(
        self,
        old_confidence: float,
        new_confidence: float,
        execution_success: bool,
        total_executions: int,
        error_magnitude: Optional[float]
    ) -> str:
        """Generate human-readable reasoning for confidence change."""
        change = new_confidence - old_confidence
        direction = "increased" if change > 0 else "decreased"
        
        reasons = []
        
        if execution_success:
            reasons.append("successful execution")
            if error_magnitude and error_magnitude < 0.05:
                reasons.append(f"low error ({error_magnitude:.1%})")
        else:
            reasons.append("failed execution")
        
        if total_executions < self.min_samples:
            reasons.append(f"limited sample size ({total_executions} executions)")
        elif total_executions > 100:
            reasons.append(f"extensive testing ({total_executions} executions)")
        
        reason_str = ", ".join(reasons)
        
        return (
            f"Confidence {direction} by {abs(change):.3f} "
            f"({old_confidence:.3f} → {new_confidence:.3f}) "
            f"due to: {reason_str}"
        )
    
    def get_learning_insights(
        self,
        db: Session,
        formula_id: Optional[int] = None,
        days: int = 7
    ) -> Dict[str, Any]:
        """
        Get learning insights and trends.
        
        Returns statistics about:
        - Confidence trends over time
        - Context-specific performance patterns
        - Auto-deployment candidates
        - Formulas needing review
        """
        since_date = datetime.utcnow() - timedelta(days=days)
        
        # Get recent learning events
        events_query = db.query(LearningEvent).filter(
            LearningEvent.timestamp >= since_date
        )
        
        if formula_id:
            events_query = events_query.filter(LearningEvent.formula_id == formula_id)
        
        events = events_query.order_by(LearningEvent.timestamp.desc()).all()
        
        # Analyze trends
        confidence_increases = sum(1 for e in events if e.new_confidence > e.old_confidence)
        confidence_decreases = sum(1 for e in events if e.new_confidence < e.old_confidence)
        
        # Get formulas ready for auto-deploy
        auto_deploy_candidates = db.query(Formula).filter(
            Formula.confidence_score >= self.auto_deploy_threshold,
            Formula.total_executions >= self.min_samples,
            Formula.status == "approved"
        ).all()
        
        # Get formulas needing review
        review_needed = db.query(Formula).filter(
            Formula.confidence_score >= self.human_review_threshold,
            Formula.confidence_score < self.auto_deploy_threshold,
            Formula.status == "pending_review"
        ).all()
        
        return {
            "period_days": days,
            "total_learning_events": len(events),
            "confidence_increases": confidence_increases,
            "confidence_decreases": confidence_decreases,
            "auto_deploy_candidates": len(auto_deploy_candidates),
            "formulas_needing_review": len(review_needed),
            "recent_events": [
                {
                    "event_id": e.event_id,
                    "formula_id": e.formula_id,
                    "event_type": e.event_type,
                    "confidence_change": e.new_confidence - e.old_confidence if e.new_confidence and e.old_confidence else None,
                    "timestamp": e.timestamp.isoformat()
                }
                for e in events[:20]
            ]
        }


# Global instance
tinker_ml = TinkerML()
