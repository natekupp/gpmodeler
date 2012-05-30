#!/usr/bin/env python

"""
(The MIT License)

Copyright (C) 2011-2012 Nathan Kupp, Yale University.

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import datetime, os, pandas, csv, argparse, fnmatch
import numpy as np
from sklearn.gaussian_process import GaussianProcess

def parse_args():
    """Parse command-line arguments to this script.
    """
    desc = """Build Gaussian process models on wafer data. Expects ./train/ and
    ./test/ subfolders of current directory to have training and test data for 
    each wafer to predict.
    """
    epilog = """Open-sourced under the MIT license.
    Copyright (C) 2011-2012 Nathan Kupp, Yale University."""
    
    parser = argparse.ArgumentParser(description=desc, epilog=epilog)
    parser.add_argument('max_x', type=int,
                        help='the wafer width dimension, in number of chips')
                        
    parser.add_argument('max_y', type=int,
                        help='the wafer height dimension, in number of chips')
                        
    parser.add_argument('--noise_param', type=float,
                        default=0.1, required=False,
                        help='noise parameter, default = 0.1')
                        
    return parser.parse_args()


def time_stamped(fname, fmt='%Y-%m-%d__%H-%M-%S_{fname}'):
    """Prefix a file name with a timestamp.
    """
    return datetime.datetime.now().strftime(fmt).format(fname=fname)


def compute_radius(X, max_x, max_y):
    """Compute the radius of a chip coordinate.
    """
    center          = [max_x / 2.0, \
                       max_y / 2.0]
    centered_coords = [X.ix[:,0] - center[0], \
                       X.ix[:,1] - center[1]]
    return np.linalg.norm(centered_coords)


def fit_and_predict(X_train, y_train, X_test, y_test):
    """Fit a gaussian process model and predict, reporting resultant percent
    error.
    """
    
    # Train Gaussian Process Model. Using noise parameter 
    # specified by the user.
    gp = GaussianProcess(nugget = args.noise_param)
    gp.fit(X_train, y_train)
    
    # Predict and record absolute percent error.
    y_pred = gp.predict(X_test)
    
    assert y_test.shape == y_pred.shape, \
        'test and predicted y column dimensions do not match.'
    
    return np.abs(np.mean((y_pred - y_test) / y_test))
    

if __name__ == "__main__":
    args         = parse_args()
    
    # Get list of training wafer files in ./train/
    train_path   = os.path.join(os.getcwd(), 'train')
    assert os.path.exists(train_path), \
        './train/ directory does not exist.'
    train_files  = os.listdir(train_path)
    train_files  = fnmatch.filter(train_files, '*.csv')

    # Get list of test wafer files in ./test/    
    test_path    = os.path.join(os.getcwd(), 'test')
    assert os.path.exists(test_path), \
        './test/ directory does not exist.'
    test_files   = os.listdir(test_path)
    test_files   = fnmatch.filter(test_files, '*.csv')
    
    assert train_files == test_files, \
        'training wafer data does not match test wafer data.'
    
    assert len(train_files) > 0, \
        'must have at least 1 wafer of data.'
        
    # Create a list of wafer names, without '.csv'
    wafer_names = [w.strip('.csv') for w in train_files]
        
    # Pull out a list of (dependent variable) column names; strip out X/Y cols, 
    # as X/Y are the independent variables.
    first_wafer = os.path.join(train_path, train_files[0])
    cols = pandas.read_csv(first_wafer).columns.tolist()
    cols = [c for c in cols if c not in ['X', 'Y']]
    
    # Create array of size {n_wafers x n_columns} to store prediction errors.
    # We index rows with wafer names.
    results = pandas.DataFrame(np.zeros((len(test_files), len(cols))), \
                               index=wafer_names, \
                               columns=cols)

    # We keep a running log of results, writing to a CSV file upon completion of
    # predictions on each wafer.
    with open(time_stamped('results.csv'), 'w') as outfile:
        c = csv.writer(outfile)
        c.writerow(['wafer'] + cols)
        
        # iterate over wafers
        for i, wafer in enumerate(train_files):
            print '[ %4d / %4d ] %15s ' % (i+1, len(train_files), wafer)
            
            try:
                # load matrices of training/test data
                train_data = pandas.read_csv(os.path.join(train_path, wafer))
                test_data  = pandas.read_csv(os.path.join(test_path, wafer))
                
                # create n_chips x {X,Y,R} predictor matrix
                X_train           = train_data[['X','Y']]
                X_train['radius'] = compute_radius(X_train, \
                                                   args.max_x, args.max_y)        
                
                # create n_chips x {X,Y,R} predictor matrix
                X_test            = test_data[['X', 'Y']]
                X_test['radius']  = compute_radius(X_test, \
                                                   args.max_x, args.max_y)        

                # iterate over dependent variable columns, create y-vector for
                # each, train and predict.
                for j, col in enumerate(cols):
                    y_train = train_data[col]
                    y_test  = test_data[col]
                    results.ix[i, j] = \
                        fit_and_predict(X_train, y_train, X_test, y_test)
                
                # save results for this wafer to results CSV file
                c.writerow([wafer] + results.ix[i, :].tolist())

            except Exception, e:
                # We don't want to stop on exceptions, just print an
                # error and continue to the next wafer.
                print e



