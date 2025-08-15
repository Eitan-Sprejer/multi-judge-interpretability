#!/bin/bash

# Quick Start Script for Multi-LLM Judge Creation Experiment
# This script helps you get started quickly

echo "🚀 Multi-LLM Judge Creation - Quick Start"
echo "=========================================="

# Check if we're in the right directory
if [ ! -f "create_judges.py" ]; then
    echo "❌ Error: Please run this script from the 3b_judge_self_bias_analysis directory"
    exit 1
fi

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python 3 is not installed or not in PATH"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source .venv/bin/activate

# Install requirements
echo "📥 Installing requirements..."
pip install -r requirements.txt

# Check environment variables
if [ -z "$MARTIAN_API_KEY" ]; then
    echo "⚠️  Warning: MARTIAN_API_KEY environment variable not set"
    echo "   Please set it before proceeding:"
    echo "   export MARTIAN_API_KEY='your-api-key-here'"
    echo ""
fi

echo "✅ Setup complete!"
echo ""
echo "📚 Next steps:"
echo "1. Set your MARTIAN_API_KEY environment variable"
echo "2. Test the setup: python3 test_setup.py"
echo "3. Create judges (quick): python3 run_experiment.py --quick"
echo "4. Create judges (full):  python3 run_experiment.py"
echo "5. List judges: python3 list_judges.py"
echo "6. See examples: python3 example_usage.py"
echo ""
echo "🎯 Goal: Create 50 judges (5 LLM providers × 10 rubric types)"
echo ""
echo "Happy judging! 🧑‍⚖️"
