# test_app.py - Comprehensive Test Suite for K-Means App
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import numpy as np
import time
from datetime import datetime
import traceback

class AppTester:
    """Comprehensive testing suite for the K-Means Customer Segmentation App"""

    def __init__(self):
        self.test_results = []
        self.start_time = None
        self.test_data_path = "Customer Data.csv"

    def load_clean_data(self):
        """Load data and clean column names"""
        df = pd.read_csv(self.test_data_path)
        df.columns = df.columns.str.strip()
        return df

    def log(self, message, level="INFO"):
        """Log test messages"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] [{level}] {message}")

    def test_result(self, test_name, passed, duration, error=None):
        """Record test result"""
        status = "[PASS]" if passed else "[FAIL]"
        self.test_results.append({
            'test': test_name,
            'passed': passed,
            'duration': duration,
            'error': error
        })
        self.log(f"{status} - {test_name} ({duration:.2f}s)", "RESULT")
        if error:
            self.log(f"Error: {error}", "ERROR")

    def run_test(self, test_func, test_name):
        """Run a single test with timing and error handling"""
        self.log(f"Running: {test_name}", "TEST")
        start = time.time()
        try:
            test_func()
            duration = time.time() - start
            self.test_result(test_name, True, duration)
            return True
        except Exception as e:
            duration = time.time() - start
            error_msg = str(e) + "\n" + traceback.format_exc()
            self.test_result(test_name, False, duration, error_msg)
            return False

    # ========== DATA LOADING TESTS ==========

    def test_1_data_file_exists(self):
        """Test 1: Check if data files exist"""
        files = ['Customer Data.csv', 'merkaba_raw.csv']
        found = [f for f in files if os.path.exists(f)]
        assert len(found) > 0, f"No data files found. Checked: {files}"
        self.log(f"Found data files: {found}")

    def test_2_load_data_basic(self):
        """Test 2: Load data with pandas"""
        df = self.load_clean_data()
        assert df is not None, "Failed to load data"
        assert len(df) > 0, "Data is empty"
        self.log(f"Loaded {len(df):,} rows, {len(df.columns)} columns")

    def test_3_required_columns(self):
        """Test 3: Validate required columns exist"""
        df = self.load_clean_data()
        required = ['CUST_ID', 'BALANCE', 'PURCHASES', 'CASH_ADVANCE', 'CREDIT_LIMIT']
        missing = [col for col in required if col not in df.columns]
        assert len(missing) == 0, f"Missing required columns: {missing}"
        self.log(f"All required columns present: {required}")

    def test_4_data_types(self):
        """Test 4: Validate data types"""
        df = self.load_clean_data()
        numeric_cols = ['BALANCE', 'PURCHASES', 'CASH_ADVANCE', 'CREDIT_LIMIT']
        for col in numeric_cols:
            if col in df.columns:
                assert pd.api.types.is_numeric_dtype(df[col]), f"{col} is not numeric"
        self.log("Data types validated successfully")

    def test_5_chunked_loading(self):
        """Test 5: Load data in chunks"""
        chunk_size = 1000
        chunks = []
        for chunk in pd.read_csv(self.test_data_path, chunksize=chunk_size):
            chunks.append(chunk)
        df = pd.concat(chunks, ignore_index=True)
        assert len(df) > 0, "Chunked loading failed"
        self.log(f"Chunked loading successful: {len(chunks)} chunks, {len(df):,} total rows")

    # ========== PREPROCESSING TESTS ==========

    def test_6_missing_values_handling(self):
        """Test 6: Handle missing values"""
        df = self.load_clean_data()
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        missing_after = df[numeric_cols].isnull().sum().sum()
        assert missing_after == 0, f"Still have {missing_after} missing values"
        self.log("Missing values handled successfully")

    def test_7_feature_selection(self):
        """Test 7: Select and validate features"""
        df = self.load_clean_data()
        recommended_features = [
            'BALANCE', 'PURCHASES', 'CASH_ADVANCE',
            'CREDIT_LIMIT', 'PAYMENTS'
        ]
        available = [f for f in recommended_features if f in df.columns]
        assert len(available) >= 3, f"Need at least 3 features, got {len(available)}"
        self.log(f"Selected {len(available)} features: {available}")

    def test_8_data_scaling(self):
        """Test 8: Standard scaling"""
        from sklearn.preprocessing import StandardScaler
        df = self.load_clean_data()
        features = ['BALANCE', 'PURCHASES', 'CASH_ADVANCE', 'CREDIT_LIMIT']
        features = [f for f in features if f in df.columns]

        X = df[features].fillna(df[features].median())
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Check scaling worked
        mean_scaled = np.abs(X_scaled.mean(axis=0)).max()
        std_scaled = np.abs(X_scaled.std(axis=0) - 1.0).max()

        assert mean_scaled < 1e-10, f"Mean not centered: {mean_scaled}"
        assert std_scaled < 1e-10, f"Std not 1: {std_scaled}"
        self.log(f"Scaling validated: mean~0, std~1")

    # ========== CLUSTERING TESTS ==========

    def test_9_kmeans_small_k(self):
        """Test 9: K-Means with K=3"""
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler

        df = self.load_clean_data()
        features = ['BALANCE', 'PURCHASES', 'CASH_ADVANCE', 'CREDIT_LIMIT']
        features = [f for f in features if f in df.columns]

        X = df[features].fillna(df[features].median())
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)

        assert len(np.unique(clusters)) == 3, f"Expected 3 clusters, got {len(np.unique(clusters))}"
        self.log(f"K-Means (K=3) successful: {len(np.unique(clusters))} clusters")

    def test_10_minibatch_kmeans(self):
        """Test 10: MiniBatch K-Means for large data"""
        from sklearn.cluster import MiniBatchKMeans
        from sklearn.preprocessing import StandardScaler

        df = self.load_clean_data()
        features = ['BALANCE', 'PURCHASES', 'CASH_ADVANCE', 'CREDIT_LIMIT']
        features = [f for f in features if f in df.columns]

        X = df[features].fillna(df[features].median())
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        kmeans = MiniBatchKMeans(n_clusters=4, random_state=42, batch_size=1000)
        clusters = kmeans.fit_predict(X_scaled)

        assert len(np.unique(clusters)) == 4, f"Expected 4 clusters, got {len(np.unique(clusters))}"
        self.log(f"MiniBatch K-Means (K=4) successful")

    def test_11_silhouette_score(self):
        """Test 11: Calculate silhouette score"""
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import silhouette_score

        df = self.load_clean_data()
        features = ['BALANCE', 'PURCHASES', 'CASH_ADVANCE', 'CREDIT_LIMIT']
        features = [f for f in features if f in df.columns]

        X = df[features].fillna(df[features].median())
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        kmeans = KMeans(n_clusters=4, random_state=42, n_init=5)
        clusters = kmeans.fit_predict(X_scaled)

        score = silhouette_score(X_scaled, clusters)
        assert -1 <= score <= 1, f"Invalid silhouette score: {score}"
        self.log(f"Silhouette score: {score:.3f}")

    def test_12_elbow_method_fast(self):
        """Test 12: Elbow method (K=2 to 5 only for speed)"""
        from sklearn.cluster import MiniBatchKMeans
        from sklearn.preprocessing import StandardScaler

        df = self.load_clean_data()
        features = ['BALANCE', 'PURCHASES', 'CASH_ADVANCE', 'CREDIT_LIMIT']
        features = [f for f in features if f in df.columns]

        X = df[features].fillna(df[features].median())
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        inertias = []
        k_range = range(2, 6)  # Only test K=2 to 5 for speed

        for k in k_range:
            kmeans = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=1000, max_iter=50)
            kmeans.fit(X_scaled)
            inertias.append(kmeans.inertia_)

        assert len(inertias) == len(k_range), "Elbow method failed"
        assert all(inertias[i] >= inertias[i+1] for i in range(len(inertias)-1)), "Inertia should decrease"
        self.log(f"Elbow method completed: {list(k_range)}")

    # ========== PERFORMANCE TESTS ==========

    def test_13_large_dataset_performance(self):
        """Test 13: Performance with full dataset"""
        from sklearn.cluster import MiniBatchKMeans
        from sklearn.preprocessing import StandardScaler

        start = time.time()

        df = self.load_clean_data()
        features = ['BALANCE', 'PURCHASES', 'CASH_ADVANCE', 'CREDIT_LIMIT']
        features = [f for f in features if f in df.columns]

        X = df[features].fillna(df[features].median())
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        kmeans = MiniBatchKMeans(n_clusters=4, random_state=42, batch_size=1000)
        clusters = kmeans.fit_predict(X_scaled)

        duration = time.time() - start

        assert duration < 10, f"Too slow: {duration:.1f}s (should be < 10s)"
        self.log(f"Full dataset clustering: {len(df):,} rows in {duration:.2f}s")

    def test_14_memory_usage(self):
        """Test 14: Memory usage check"""
        df = self.load_clean_data()
        memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024

        assert memory_mb < 100, f"Memory usage too high: {memory_mb:.2f} MB"
        self.log(f"Memory usage: {memory_mb:.2f} MB")

    # ========== MODEL PERSISTENCE TESTS ==========

    def test_15_save_load_model(self):
        """Test 15: Save and load model"""
        import pickle
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler

        # Train a model
        df = self.load_clean_data()
        features = ['BALANCE', 'PURCHASES', 'CASH_ADVANCE', 'CREDIT_LIMIT']
        features = [f for f in features if f in df.columns]

        X = df[features].fillna(df[features].median())
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        kmeans = KMeans(n_clusters=4, random_state=42, n_init=5)
        kmeans.fit(X_scaled)

        # Save model
        os.makedirs('models', exist_ok=True)
        model_path = 'models/test_model.pkl'

        with open(model_path, 'wb') as f:
            pickle.dump({'model': kmeans, 'scaler': scaler}, f)

        # Load model
        with open(model_path, 'rb') as f:
            loaded = pickle.load(f)

        # Verify
        predictions_orig = kmeans.predict(X_scaled[:10])
        predictions_loaded = loaded['model'].predict(X_scaled[:10])

        assert np.array_equal(predictions_orig, predictions_loaded), "Model predictions differ"
        self.log("Model save/load successful")

        # Cleanup
        if os.path.exists(model_path):
            os.remove(model_path)

    # ========== RUN ALL TESTS ==========

    def run_all_tests(self):
        """Run all tests"""
        self.log("=" * 60, "INFO")
        self.log("K-MEANS APP COMPREHENSIVE TEST SUITE", "INFO")
        self.log("=" * 60, "INFO")

        self.start_time = time.time()

        tests = [
            (self.test_1_data_file_exists, "Data Files Exist"),
            (self.test_2_load_data_basic, "Load Data (Basic)"),
            (self.test_3_required_columns, "Required Columns"),
            (self.test_4_data_types, "Data Types Validation"),
            (self.test_5_chunked_loading, "Chunked Data Loading"),
            (self.test_6_missing_values_handling, "Missing Values Handling"),
            (self.test_7_feature_selection, "Feature Selection"),
            (self.test_8_data_scaling, "Data Scaling (StandardScaler)"),
            (self.test_9_kmeans_small_k, "K-Means (K=3)"),
            (self.test_10_minibatch_kmeans, "MiniBatch K-Means (K=4)"),
            (self.test_11_silhouette_score, "Silhouette Score"),
            (self.test_12_elbow_method_fast, "Elbow Method (Fast)"),
            (self.test_13_large_dataset_performance, "Large Dataset Performance"),
            (self.test_14_memory_usage, "Memory Usage Check"),
            (self.test_15_save_load_model, "Model Save/Load"),
        ]

        passed = 0
        failed = 0

        for test_func, test_name in tests:
            if self.run_test(test_func, test_name):
                passed += 1
            else:
                failed += 1
            print()  # Empty line between tests

        total_duration = time.time() - self.start_time

        # Print summary
        self.log("=" * 60, "INFO")
        self.log("TEST SUMMARY", "INFO")
        self.log("=" * 60, "INFO")
        self.log(f"Total Tests: {len(tests)}", "INFO")
        self.log(f"Passed: {passed}", "INFO")
        self.log(f"Failed: {failed}", "INFO")
        self.log(f"Success Rate: {passed/len(tests)*100:.1f}%", "INFO")
        self.log(f"Total Duration: {total_duration:.2f}s", "INFO")
        self.log("=" * 60, "INFO")

        # Detailed results
        print("\nDetailed Results:")
        print("-" * 80)
        print(f"{'Test Name':<45} {'Status':<10} {'Duration':<15}")
        print("-" * 80)

        for result in self.test_results:
            status = "[PASS]" if result['passed'] else "[FAIL]"
            print(f"{result['test']:<45} {status:<10} {result['duration']:.2f}s")

        print("-" * 80)

        return passed == len(tests)

if __name__ == "__main__":
    tester = AppTester()
    success = tester.run_all_tests()
    sys.exit(0 if success else 1)
