"""
Command-line interface for GMM-CT.

This module provides a CLI for running GMM-CT reconstruction from the terminal.
"""

import argparse
import sys
from pathlib import Path


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="GMM-CT: Gaussian Mixture Model CT Reconstruction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version='%(prog)s 0.1.0'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Reconstruct command
    reconstruct_parser = subparsers.add_parser(
        'reconstruct',
        help='Run GMM reconstruction on projection data'
    )
    reconstruct_parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to input projection data (.npy or .pt file)'
    )
    reconstruct_parser.add_argument(
        '--output',
        type=str,
        default='results/',
        help='Output directory for results'
    )
    reconstruct_parser.add_argument(
        '--n-gaussians',
        type=int,
        default=3,
        help='Number of Gaussian components'
    )
    reconstruct_parser.add_argument(
        '--device',
        type=str,
        choices=['cpu', 'cuda'],
        default='cpu',
        help='Computation device'
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return 0
    
    if args.command == 'reconstruct':
        print(f"GMM-CT Reconstruction")
        print(f"Input: {args.input}")
        print(f"Output: {args.output}")
        print(f"Number of Gaussians: {args.n_gaussians}")
        print(f"Device: {args.device}")
        print("\n⚠️  Full CLI implementation coming soon!")
        print("For now, please use the Python API directly.")
        print("See: docs/guides/quickstart.md")
        return 0
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
