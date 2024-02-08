# MIT License
# 
# Copyright (c) 2024 Romain Ilbert
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os


def configure_environment():
    """Configures the environment settings for TensorFlow and directories."""
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TensorFlow logging
    current_directory = os.getcwd()
    return current_directory

def setup_experiment_id(args):
    """
    Constructs an experiment identifier string using the provided command line arguments.
    The ID format incorporates key experiment settings for easy reference.

    Parameters:
        args (argparse.Namespace): Command line arguments specifying experiment settings.

    Returns:
        str: A unique identifier for the experiment, incorporating key settings.
    """
    # Basic experiment identifier, common for all models
    base_id = f"{args.data}_{args.feature_type}_{args.model}_sl{args.seq_len}_pl{args.pred_len}_lr{args.learning_rate}"

    # Additional settings for specific models or conditions
    if args.model in ['transformer', 'transformer_random', 'spectrans']:
        sam_suffix = "_sam" if args.use_sam else ""
        extra = f"_heads_{args.num_heads}_d_model_{args.d_model}{sam_suffix}"
    elif args.model == 'linear':
        extra = ""
    elif args.model == 'tsmixer':  
        extra = f"_nb{args.n_block}_dp{args.dropout}_fd{args.ff_dim}"
    else :
        raise ValueError(f'Unknown model type: {args.model}')
    
    # Norm type and activation are common additions not dependent on the model type
    common_suffix = f"_{extra}"

    return base_id + common_suffix