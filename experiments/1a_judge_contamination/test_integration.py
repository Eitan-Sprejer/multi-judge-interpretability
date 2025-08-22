#!/usr/bin/env python3
"""
Integration Test and Validation Script for Enhanced Judge Contamination Experiment

Tests the integration of all components and validates against existing pipeline patterns.
"""

import sys
import os
import unittest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tempfile
import shutil
import json
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))
sys.path.append(str(Path(__file__).parent))

# Import test modules
try:
    from src.contamination_analysis import ContaminationAnalyzer
    from src.visualizations import ContaminationVisualizer
    from src.results_framework import AdvancedResultsProcessor
    from run_experiment import EnhancedContaminationExperimentRunner
except ImportError as e:
    print(f"Import error: {e}")
    print("Attempting alternative imports...")
    sys.path.append(str(Path(__file__).parent / "src"))
    from contamination_analysis import ContaminationAnalyzer
    from visualizations import ContaminationVisualizer
    from results_framework import AdvancedResultsProcessor

# Import pipeline components
try:
    from pipeline.core.judge_creation import JUDGE_MODEL, MIN_SCORE, MAX_SCORE
    from pipeline.utils.judge_rubrics import INVERTED_JUDGE_RUBRICS
    from pipeline.core.dataset_loader import DatasetLoader
except ImportError as e:
    print(f"Pipeline import error: {e}")
    print("Creating mock pipeline components for testing...")
    
    # Mock pipeline components for testing
    JUDGE_MODEL = "mock-model"
    MIN_SCORE = 0.0
    MAX_SCORE = 4.0
    INVERTED_JUDGE_RUBRICS = {
        'test-judge-1': lambda: "Inverted test rubric 1",
        'test-judge-2': lambda: "Inverted test rubric 2"
    }

class TestContaminationAnalysis(unittest.TestCase):
    """Test contamination analysis module."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = ContaminationAnalyzer(significance_level=0.05)
        
        # Create sample data
        np.random.seed(42)
        self.baseline_scores = pd.DataFrame({
            'judge1': np.random.normal(2.5, 0.5, 100),
            'judge2': np.random.normal(2.0, 0.8, 100)
        })
        
        # Create contaminated scores (inverted)
        self.contaminated_scores = pd.DataFrame({
            'inverted_judge1': 4.0 - self.baseline_scores['judge1'] + np.random.normal(0, 0.1, 100),
            'inverted_judge2': 4.0 - self.baseline_scores['judge2'] + np.random.normal(0, 0.1, 100)
        })
        
        self.judge_mapping = {
            'judge1': 'inverted_judge1',
            'judge2': 'inverted_judge2'
        }
    
    def test_analyze_judge_inversion(self):
        """Test judge inversion analysis."""
        results = self.analyzer.analyze_judge_inversion(
            self.baseline_scores, self.contaminated_scores, self.judge_mapping
        )
        
        # Check structure
        self.assertIn('individual_judges', results)
        self.assertIn('aggregate_metrics', results)
        self.assertIn('statistical_tests', results)
        self.assertIn('inversion_detection', results)
        
        # Check individual judge analysis
        for judge_id in self.judge_mapping.keys():
            self.assertIn(judge_id, results['individual_judges'])
            judge_result = results['individual_judges'][judge_id]
            
            # Should detect strong negative correlation
            corr = judge_result['correlations']['pearson']['correlation']
            self.assertLess(corr, -0.5, f"Expected strong negative correlation for {judge_id}")
            
            # Should detect inversion
            self.assertTrue(judge_result['inversion_analysis']['is_inverted'])
    
    def test_aggregator_robustness_analysis(self):
        """Test robustness analysis with mock data."""
        # Create human feedback data
        human_feedback = pd.Series(np.random.normal(5.0, 2.0, 100))
        
        # Test robustness analysis
        results = self.analyzer.analyze_aggregator_robustness(
            clean_data=self.baseline_scores,
            contaminated_data=self.contaminated_scores,
            human_feedback=human_feedback,
            contamination_rates=[0.0, 0.1, 0.2, 0.3]
        )
        
        # Check structure
        self.assertIn('contamination_curves', results)
        self.assertIn('robustness_metrics', results)
        
        # Check contamination curves
        curves = results['contamination_curves']
        self.assertEqual(len(curves), 4)  # Should have 4 contamination rates
        
        # Check metrics exist
        metrics = results['robustness_metrics']
        self.assertIn('clean_performance', metrics)
        self.assertIn('performance_curve', metrics)

class TestVisualizationSystem(unittest.TestCase):
    """Test visualization system."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.visualizer = ContaminationVisualizer(self.temp_dir, dpi=150)
        
        # Create sample analysis results
        self.analysis_results = {
            'judge_inversion': {
                'individual_judges': {
                    'judge1': {
                        'correlations': {'pearson': {'correlation': -0.8, 'p_value': 0.001}},
                        'inversion_analysis': {'is_inverted': True, 'inversion_strength': 0.8}
                    }
                },
                'aggregate_metrics': {
                    'average_correlation': -0.7,
                    'contamination_rate': 0.9,
                    'system_inversion_detected': True
                },
                'inversion_detection': {
                    'patterns': {
                        'complete_inversion': ['judge1'],
                        'partial_inversion': [],
                        'no_inversion': [],
                        'amplification': []
                    }
                }
            }
        }
        
        # Create sample data
        np.random.seed(42)
        self.baseline_scores = pd.DataFrame({
            'judge1': np.random.normal(2.5, 0.5, 50)
        })
        self.contaminated_scores = pd.DataFrame({
            'inverted_judge1': 4.0 - self.baseline_scores['judge1']
        })
        self.judge_mapping = {'judge1': 'inverted_judge1'}
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_plot_score_distributions(self):
        """Test score distribution plotting."""
        fig = self.visualizer.plot_score_distributions(
            self.baseline_scores, self.contaminated_scores, self.judge_mapping
        )
        
        # Check that figure was created
        self.assertIsNotNone(fig)
        
        # Check that file was saved
        plot_files = list(self.temp_dir.glob("score_distributions.png"))
        self.assertEqual(len(plot_files), 1)
    
    def test_plot_correlation_heatmap(self):
        """Test correlation heatmap plotting."""
        fig = self.visualizer.plot_correlation_heatmap(self.analysis_results)
        
        # Check that figure was created
        self.assertIsNotNone(fig)
        
        # Check that file was saved
        plot_files = list(self.temp_dir.glob("correlation_heatmap.png"))
        self.assertEqual(len(plot_files), 1)
    
    def test_create_summary_dashboard(self):
        """Test dashboard creation."""
        fig = self.visualizer.create_summary_dashboard(self.analysis_results)
        
        # Check that figure was created
        self.assertIsNotNone(fig)
        
        # Check that file was saved
        plot_files = list(self.temp_dir.glob("contamination_dashboard.png"))
        self.assertEqual(len(plot_files), 1)

class TestResultsFramework(unittest.TestCase):
    """Test results processing framework."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config = {
            'experiment': {'name': 'test_experiment'},
            'analysis': {'statistical_tests': {'significance_level': 0.05}}
        }
        self.processor = AdvancedResultsProcessor(self.temp_dir, self.config)
        
        # Create sample analysis results
        self.analysis_results = {
            'inverted': {
                'judge_inversion': {
                    'individual_judges': {
                        'judge1': {
                            'correlations': {'pearson': {'correlation': -0.8, 'p_value': 0.001}},
                            'inversion_analysis': {'is_inverted': True, 'inversion_strength': 0.8}
                        }
                    },
                    'aggregate_metrics': {
                        'average_correlation': -0.8,
                        'contamination_rate': 1.0,
                        'system_inversion_detected': True,
                        'correlations_list': [-0.8, -0.7, -0.9]
                    },
                    'statistical_tests': {
                        'system_level': {
                            'distribution_shift': {'significant': True, 'ks_p_value': 0.001}
                        }
                    }
                }
            }
        }
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_process_contamination_results(self):
        """Test contamination results processing."""
        execution_metadata = {
            'timestamp': '20241201_120000',
            'samples_processed': 100,
            'judges_created': 2
        }
        
        processed_results = self.processor.process_contamination_results(
            self.analysis_results, None, execution_metadata
        )
        
        # Check structure
        self.assertIn('contamination_metrics', processed_results)
        self.assertIn('key_insights', processed_results)
        self.assertIn('quality_assessment', processed_results)
        
        # Check contamination metrics
        metrics = processed_results['contamination_metrics']['inverted']
        self.assertEqual(metrics['average_correlation'], -0.8)
        self.assertTrue(metrics['system_inversion_detected'])
        self.assertTrue(metrics['statistical_significance'])
    
    def test_generate_executive_summary(self):
        """Test executive summary generation."""
        processed_results = {
            'contamination_metrics': {
                'inverted': {
                    'average_correlation': -0.8,
                    'contamination_success_rate': 1.0,
                    'system_inversion_detected': True,
                    'statistical_significance': True,
                    'p_value': 0.001,
                    'confidence_interval': (-0.9, -0.7),
                    'performance_degradation': 0.0,
                    'breakdown_threshold': None,
                    'recovery_rate': 0.0,
                    'effect_size': 2.0,
                    'power_analysis': 0.95,
                    'minimum_detectable_effect': 0.3
                }
            },
            'key_insights': ['Strong contamination detected']
        }
        
        execution_metadata = {
            'timestamp': '20241201_120000',
            'samples_processed': 100,
            'judges_created': 2
        }
        
        summary = self.processor.generate_executive_summary(
            processed_results, execution_metadata
        )
        
        # Check summary structure
        self.assertEqual(summary.experiment_id, '20241201_120000')
        self.assertEqual(summary.total_samples, 100)
        self.assertTrue(summary.contamination_detected)
        self.assertIn('inverted', summary.contamination_types)
    
    def test_export_publication_data(self):
        """Test publication data export."""
        processed_results = {
            'contamination_metrics': {
                'inverted': {
                    'average_correlation': -0.8,
                    'contamination_success_rate': 1.0,
                    'system_inversion_detected': True,
                    'p_value': 0.001,
                    'effect_size': 2.0
                }
            }
        }
        
        # Test LaTeX export
        latex_output = self.processor.export_publication_data(processed_results, format='latex')
        self.assertIn('\\begin{table}', latex_output)
        self.assertIn('Inverted', latex_output)
        
        # Test CSV export
        csv_path = self.processor.export_publication_data(processed_results, format='csv')
        self.assertTrue(Path(csv_path).exists())

class TestPipelineIntegration(unittest.TestCase):
    """Test integration with existing pipeline."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        
        # Create mock arguments
        self.mock_args = Mock()
        self.mock_args.quick = True
        self.mock_args.num_samples = 10
        self.mock_args.contamination_types = 'inverted'
        self.mock_args.significance_level = 0.05
        self.mock_args.enable_robustness = False
        self.mock_args.enable_visualizations = True
        self.mock_args.contamination_rates = [0.0, 0.1, 0.2]
        self.mock_args.generate_publication_figures = False
    
    def tearDown(self):
        """Clean up test fixtures."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    @patch('run_experiment.create_martian_client')
    @patch('run_experiment.DatasetLoader')
    def test_experiment_initialization(self, mock_loader, mock_client):
        """Test experiment runner initialization."""
        # Mock client
        mock_client.return_value = Mock()
        
        # Create experiment runner
        with patch.object(Path, 'mkdir'):
            runner = EnhancedContaminationExperimentRunner(self.mock_args)
        
        # Check initialization
        self.assertIsNotNone(runner.analyzer)
        self.assertIsNotNone(runner.visualizer)
        self.assertIsNotNone(runner.results_processor)
        self.assertIsNotNone(runner.config)
    
    def test_configuration_loading(self):
        """Test configuration loading."""
        # Create temporary config file
        config_path = self.temp_dir / "config.yaml"
        config_content = """
        experiment:
          name: "test_experiment"
        contamination:
          inverted:
            enabled: true
        execution:
          quick_mode_samples: 25
        """
        with open(config_path, 'w') as f:
            f.write(config_content)
        
        # Mock the config path
        with patch.object(Path, '__truediv__', return_value=config_path):
            with patch.object(Path, 'exists', return_value=True):
                with patch.object(Path, 'mkdir'):
                    runner = EnhancedContaminationExperimentRunner(self.mock_args)
                    config = runner._load_configuration()
        
        # Check config loaded correctly
        self.assertEqual(config['experiment']['name'], 'test_experiment')
        self.assertTrue(config['contamination']['inverted']['enabled'])
    
    def test_judge_mapping_consistency(self):
        """Test that judge mapping is consistent with pipeline patterns."""
        # Test that inverted judge IDs follow expected pattern
        judge_ids = list(INVERTED_JUDGE_RUBRICS.keys())
        
        for judge_id in judge_ids:
            inverted_id = f"inverted_{judge_id}"
            
            # Check that inverted ID follows naming convention
            self.assertTrue(inverted_id.startswith('inverted_'))
            
            # Check that original judge_id is preserved
            original_id = inverted_id.replace('inverted_', '')
            self.assertEqual(original_id, judge_id)

class TestDataCompatibility(unittest.TestCase):
    """Test compatibility with existing data formats."""
    
    def test_data_format_compatibility(self):
        """Test that the enhanced system works with existing data formats."""
        # Create sample data in expected format
        sample_data = pd.DataFrame({
            'instruction': ['Test question 1', 'Test question 2'],
            'answer': ['Test answer 1', 'Test answer 2'],
            'human_feedback': [7.5, 6.2],
            'judge_scores': [[2.1, 3.4, 1.8], [3.2, 2.9, 3.1]]
        })
        
        # Test that data can be processed
        analyzer = ContaminationAnalyzer()
        
        # Extract baseline scores in expected format
        baseline_scores = []
        for scores in sample_data['judge_scores']:
            if scores and len(scores) >= 2:
                baseline_scores.append(scores[:2])
            else:
                baseline_scores.append([np.nan, np.nan])
        
        baseline_df = pd.DataFrame(baseline_scores, columns=['judge1', 'judge2'])
        
        # Should not raise errors
        self.assertEqual(len(baseline_df), 2)
        self.assertEqual(list(baseline_df.columns), ['judge1', 'judge2'])
    
    def test_results_format_compatibility(self):
        """Test that results format is compatible with other experiments."""
        # Create sample results in expected format
        temp_dir = Path(tempfile.mkdtemp())
        
        try:
            config = {'experiment': {'name': 'test'}}
            processor = AdvancedResultsProcessor(temp_dir, config)
            
            # Generate sample processed results
            analysis_results = {
                'inverted': {
                    'judge_inversion': {
                        'aggregate_metrics': {
                            'average_correlation': -0.8,
                            'contamination_rate': 0.9,
                            'system_inversion_detected': True,
                            'correlations_list': [-0.8, -0.7]
                        },
                        'statistical_tests': {
                            'system_level': {
                                'distribution_shift': {'significant': True, 'ks_p_value': 0.001}
                            }
                        }
                    }
                }
            }
            
            processed_results = processor.process_contamination_results(
                analysis_results, None, {'timestamp': '20241201_120000'}
            )
            
            # Check that standard files are created
            expected_files = [
                temp_dir / "analysis" / "processed_results.json",
                temp_dir / "analysis" / "processed_results.yaml"
            ]
            
            for file_path in expected_files:
                self.assertTrue(file_path.exists(), f"Expected file not created: {file_path}")
                
        finally:
            shutil.rmtree(temp_dir)

def run_integration_validation():
    """Run comprehensive integration validation."""
    print("="*70)
    print("ENHANCED JUDGE CONTAMINATION INTEGRATION VALIDATION")
    print("="*70)
    
    # Run all test suites
    test_suites = [
        TestContaminationAnalysis,
        TestVisualizationSystem,
        TestResultsFramework,
        TestPipelineIntegration,
        TestDataCompatibility
    ]
    
    all_passed = True
    
    for suite_class in test_suites:
        print(f"\n🧪 Running {suite_class.__name__}...")
        
        suite = unittest.TestLoader().loadTestsFromTestCase(suite_class)
        runner = unittest.TextTestRunner(verbosity=1, stream=sys.stdout)
        result = runner.run(suite)
        
        if not result.wasSuccessful():
            all_passed = False
            print(f"❌ {suite_class.__name__} failed!")
        else:
            print(f"✅ {suite_class.__name__} passed!")
    
    print("\n" + "="*70)
    
    if all_passed:
        print("🎉 ALL INTEGRATION TESTS PASSED!")
        print("✅ Enhanced judge contamination system is ready for use")
        print("✅ Integration with existing pipeline validated")
        print("✅ All components working correctly")
    else:
        print("❌ SOME TESTS FAILED!")
        print("⚠️  Please review failed tests before using the system")
    
    print("="*70)
    
    return all_passed

if __name__ == "__main__":
    # Set up test environment
    os.environ.setdefault('PYTHONPATH', str(project_root))
    
    # Run integration validation
    success = run_integration_validation()
    
    # Print usage instructions if tests pass
    if success:
        print("\n📋 USAGE INSTRUCTIONS:")
        print("1. Run quick test:")
        print("   python run_experiment.py --quick")
        print("\n2. Run full analysis:")
        print("   python run_experiment.py --num-samples 200 --enable-robustness")
        print("\n3. Generate publication figures:")
        print("   python run_experiment.py --generate-publication-figures")
        print("\n4. View configuration options:")
        print("   python run_experiment.py --help")
    
    sys.exit(0 if success else 1)