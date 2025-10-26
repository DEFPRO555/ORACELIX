"""
Comprehensive Test Suite for Advanced Clustering Pipeline
Tests all models, metrics, and functionality
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
import tempfile
import os
import shutil

# Import our advanced clustering module
import sys
sys.path.append('..')
from models.advanced_clustering import AdvancedClusteringPipeline, create_model_manifest


class TestAdvancedClusteringPipeline:
    """Test suite for AdvancedClusteringPipeline"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing"""
        X, y = make_blobs(n_samples=1000, centers=4, n_features=8, 
                         random_state=42, cluster_std=1.5)
        df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(8)])
        return df
    
    @pytest.fixture
    def pipeline(self):
        """Create pipeline instance for testing"""
        return AdvancedClusteringPipeline(random_state=42)
    
    def test_pipeline_initialization(self, pipeline):
        """Test pipeline initialization"""
        assert pipeline.random_state == 42
        assert pipeline.models == {}
        assert pipeline.results == {}
        assert pipeline.best_model is None
        assert pipeline.best_score == -np.inf
    
    def test_prepare_data(self, pipeline, sample_data):
        """Test data preparation and scaling"""
        X_scaled = pipeline.prepare_data(sample_data)
        
        # Check data shape
        assert X_scaled.shape == sample_data.shape
        
        # Check scaling (mean should be ~0, std should be ~1)
        assert np.allclose(X_scaled.mean(axis=0), 0, atol=1e-10)
        assert np.allclose(X_scaled.std(axis=0), 1, atol=1e-10)
    
    def test_prepare_data_with_missing_values(self, pipeline):
        """Test data preparation with missing values"""
        # Create data with missing values
        X = pd.DataFrame({
            'feature_1': [1, 2, np.nan, 4, 5],
            'feature_2': [2, np.nan, 6, 8, 10],
            'feature_3': [3, 6, 9, 12, 15]
        })
        
        X_scaled = pipeline.prepare_data(X)
        
        # Should not have any NaN values
        assert not np.isnan(X_scaled).any()
        assert X_scaled.shape == (5, 3)
    
    def test_kmeans_training(self, pipeline, sample_data):
        """Test KMeans training"""
        X_scaled = pipeline.prepare_data(sample_data)
        results = pipeline.train_kmeans(X_scaled, k_range=range(2, 6))
        
        # Check results structure
        assert len(results) == 4  # k=2,3,4,5
        for k, result in results.items():
            assert 'model' in result
            assert 'labels' in result
            assert 'silhouette' in result
            assert 'davies_bouldin' in result
            assert 'calinski_harabasz' in result
            assert 'inertia' in result
            
            # Check labels
            assert len(result['labels']) == len(X_scaled)
            assert len(set(result['labels'])) == k
    
    def test_gaussian_mixture_training(self, pipeline, sample_data):
        """Test Gaussian Mixture training"""
        X_scaled = pipeline.prepare_data(sample_data)
        results = pipeline.train_gaussian_mixture(X_scaled, k_range=range(2, 6))
        
        # Check results structure
        assert len(results) == 4
        for k, result in results.items():
            assert 'model' in result
            assert 'labels' in result
            assert 'silhouette' in result
            assert 'davies_bouldin' in result
            assert 'calinski_harabasz' in result
            assert 'aic' in result
            assert 'bic' in result
    
    def test_dbscan_training(self, pipeline, sample_data):
        """Test DBSCAN training"""
        X_scaled = pipeline.prepare_data(sample_data)
        results = pipeline.train_dbscan(X_scaled, eps_range=[0.5, 1.0], min_samples_range=[2, 5])
        
        # Check that we have some results
        assert len(results) > 0
        
        for config, result in results.items():
            assert 'model' in result
            assert 'labels' in result
            assert 'silhouette' in result
            assert 'n_clusters' in result
            assert 'n_noise' in result
    
    def test_agglomerative_training(self, pipeline, sample_data):
        """Test Agglomerative Clustering training"""
        X_scaled = pipeline.prepare_data(sample_data)
        results = pipeline.train_agglomerative(X_scaled, k_range=range(2, 6), linkage=['ward', 'complete'])
        
        # Check results structure
        assert len(results) >= 8  # 4 k values * 2 linkage types
        
        for config, result in results.items():
            assert 'model' in result
            assert 'labels' in result
            assert 'silhouette' in result
            assert 'davies_bouldin' in result
            assert 'calinski_harabasz' in result
    
    def test_evaluate_all_models(self, pipeline, sample_data):
        """Test evaluation of all models"""
        results = pipeline.evaluate_all_models(sample_data)
        
        # Check that all algorithms are present
        assert 'kmeans' in results
        assert 'gaussian_mixture' in results
        assert 'dbscan' in results
        assert 'agglomerative' in results
        
        # Check that best model is selected
        assert pipeline.best_model is not None
        assert pipeline.best_score > -np.inf
    
    def test_model_comparison_table(self, pipeline, sample_data):
        """Test model comparison table generation"""
        pipeline.evaluate_all_models(sample_data)
        comparison_df = pipeline.get_model_comparison_table()
        
        # Check table structure
        assert 'Algorithm' in comparison_df.columns
        assert 'Configuration' in comparison_df.columns
        assert 'Silhouette' in comparison_df.columns
        assert 'Davies-Bouldin' in comparison_df.columns
        assert 'Calinski-Harabasz' in comparison_df.columns
        
        # Check that it's sorted by silhouette score
        assert comparison_df['Silhouette'].is_monotonic_decreasing
    
    def test_save_and_load_models(self, pipeline, sample_data):
        """Test model saving and loading"""
        # Train models
        pipeline.evaluate_all_models(sample_data)
        
        # Create temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save models
            version_dir = pipeline.save_models(temp_dir)
            assert os.path.exists(version_dir)
            
            # Check that files were created
            files = os.listdir(version_dir)
            assert 'scaler.joblib' in files
            assert 'metadata.json' in files
            
            # Create new pipeline and load models
            new_pipeline = AdvancedClusteringPipeline()
            new_pipeline.load_models(version_dir)
            
            # Check that models were loaded
            assert new_pipeline.results != {}
            assert new_pipeline.best_model is not None
            assert new_pipeline.best_score > -np.inf
    
    def test_metrics_calculation(self, pipeline, sample_data):
        """Test that all metrics are calculated correctly"""
        X_scaled = pipeline.prepare_data(sample_data)
        results = pipeline.train_kmeans(X_scaled, k_range=[3])
        
        result = results[3]
        
        # Check metric ranges
        assert 0 <= result['silhouette'] <= 1
        assert result['davies_bouldin'] >= 0
        assert result['calinski_harabasz'] >= 0
    
    def test_best_model_selection(self, pipeline, sample_data):
        """Test best model selection logic"""
        pipeline.evaluate_all_models(sample_data)
        
        # Check that best model has highest silhouette score
        best_score = pipeline.best_score
        best_algorithm = pipeline.best_model['algorithm']
        best_config = pipeline.best_model['config']
        
        # Verify this is actually the best
        best_result = pipeline.results[best_algorithm][best_config]
        assert best_result['silhouette'] == best_score
        
        # Check that no other model has higher score
        for algorithm, results in pipeline.results.items():
            for config, result in results.items():
                assert result['silhouette'] <= best_score


class TestModelManifest:
    """Test model manifest creation"""
    
    def test_create_manifest(self):
        """Test manifest creation"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create some test files
            test_file1 = os.path.join(temp_dir, "model1.joblib")
            test_file2 = os.path.join(temp_dir, "model2.joblib")
            
            with open(test_file1, 'w') as f:
                f.write("test content 1")
            with open(test_file2, 'w') as f:
                f.write("test content 2")
            
            # Create manifest
            manifest_file = create_model_manifest(temp_dir)
            
            # Check manifest was created
            assert os.path.exists(manifest_file)
            
            # Check manifest content
            import json
            with open(manifest_file, 'r') as f:
                manifest = json.load(f)
            
            assert 'model1.joblib' in manifest
            assert 'model2.joblib' in manifest
            # manifest.sha256 should NOT be in the manifest (it excludes itself)


def test_integration_workflow():
    """Integration test for complete workflow"""
    # Create sample data
    X, y = make_blobs(n_samples=500, centers=3, n_features=5, random_state=42)
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(5)])
    
    # Create pipeline
    pipeline = AdvancedClusteringPipeline(random_state=42)
    
    # Run complete workflow
    results = pipeline.evaluate_all_models(df)
    
    # Verify results
    assert pipeline.best_model is not None
    assert pipeline.best_score > -np.inf
    
    # Test comparison table
    comparison_df = pipeline.get_model_comparison_table()
    assert len(comparison_df) > 0
    
    # Test saving (in temporary directory)
    with tempfile.TemporaryDirectory() as temp_dir:
        version_dir = pipeline.save_models(temp_dir)
        assert os.path.exists(version_dir)
        
        # Test loading
        new_pipeline = AdvancedClusteringPipeline()
        new_pipeline.load_models(version_dir)
        assert new_pipeline.best_model is not None


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
