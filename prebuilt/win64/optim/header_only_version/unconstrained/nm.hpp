/*################################################################################
  ##
  ##   Copyright (C) 2016-2018 Keith O'Hara
  ##
  ##   This file is part of the OptimLib C++ library.
  ##
  ##   Licensed under the Apache License, Version 2.0 (the "License");
  ##   you may not use this file except in compliance with the License.
  ##   You may obtain a copy of the License at
  ##
  ##       http://www.apache.org/licenses/LICENSE-2.0
  ##
  ##   Unless required by applicable law or agreed to in writing, software
  ##   distributed under the License is distributed on an "AS IS" BASIS,
  ##   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  ##   See the License for the specific language governing permissions and
  ##   limitations under the License.
  ##
  ################################################################################*/

/*
 * Nelder-Mead
 */

#ifndef _optim_nm_HPP
#define _optim_nm_HPP

bool nm_int(arma::vec& init_out_vals, std::function<double (const arma::vec& vals_inp, arma::vec* grad_out, void* opt_data)> opt_objfn, void* opt_data, algo_settings_t* settings_inp);

bool nm(arma::vec& init_out_vals, std::function<double (const arma::vec& vals_inp, arma::vec* grad_out, void* opt_data)> opt_objfn, void* opt_data);
bool nm(arma::vec& init_out_vals, std::function<double (const arma::vec& vals_inp, arma::vec* grad_out, void* opt_data)> opt_objfn, void* opt_data, algo_settings_t& settings);

//

inline
bool
nm_int(arma::vec& init_out_vals, std::function<double (const arma::vec& vals_inp, arma::vec* grad_out, void* opt_data)> opt_objfn, void* opt_data, algo_settings_t* settings_inp)
{
    bool success = false;

    const size_t n_vals = init_out_vals.n_elem;

    //
    // NM settings

    algo_settings_t settings;

    if (settings_inp) {
        settings = *settings_inp;
    }

    int verbose_print_level = settings.verbose_print_level;
    
    const uint_t conv_failure_switch = settings.conv_failure_switch;
    const uint_t iter_max = settings.iter_max;
    const double err_tol = settings.err_tol;

    // expansion / contraction parameters
    const double par_alpha = settings.nm_par_alpha;
    const double par_beta  = (settings.nm_adaptive) ? 0.75 - 1.0 / (2.0*n_vals) : settings.nm_par_beta;
    const double par_gamma = (settings.nm_adaptive) ? 1.0 + 2.0 / n_vals        : settings.nm_par_gamma;
    const double par_delta = (settings.nm_adaptive) ? 1.0 - 1.0 / n_vals        : settings.nm_par_delta;

    const bool vals_bound = settings.vals_bound;
    
    const arma::vec lower_bounds = settings.lower_bounds;
    const arma::vec upper_bounds = settings.upper_bounds;

    const arma::uvec bounds_type = determine_bounds_type(vals_bound, n_vals, lower_bounds, upper_bounds);

    // lambda function for box constraints

    std::function<double (const arma::vec& vals_inp, arma::vec* grad_out, void* box_data)> box_objfn \
    = [opt_objfn, vals_bound, bounds_type, lower_bounds, upper_bounds] (const arma::vec& vals_inp, arma::vec* grad_out, void* opt_data) \
    -> double 
    {
        if (vals_bound)
        {
            arma::vec vals_inv_trans = inv_transform(vals_inp, bounds_type, lower_bounds, upper_bounds);
            
            return opt_objfn(vals_inv_trans,nullptr,opt_data);
        }
        else
        {
            return opt_objfn(vals_inp,nullptr,opt_data);
        }
    };
    
    //
    // setup

    arma::vec simplex_fn_vals(n_vals+1);
    arma::mat simplex_points(n_vals+1,n_vals);
    
    simplex_fn_vals(0) = opt_objfn(init_out_vals,nullptr,opt_data);
    simplex_points.row(0) = init_out_vals.t();

    // for (size_t i=1; i < n_vals + 1; i++) {
    //     simplex_points.row(i) = init_out_vals.t() + 0.05*arma::trans(unit_vec(i-1,n_vals));
    //     simplex_fn_vals(i) = opt_objfn(simplex_points.row(i).t(),nullptr,opt_data);
    // }

    for (size_t i=1; i < n_vals + 1; i++) 
    {
        if (init_out_vals(i-1) != 0.0) {
            simplex_points.row(i) = init_out_vals.t() + 0.05*init_out_vals(i-1)*arma::trans(unit_vec(i-1,n_vals));
        } else {
            simplex_points.row(i) = init_out_vals.t() + 0.00025*arma::trans(unit_vec(i-1,n_vals));
            // simplex_points.row(i) = init_out_vals.t() + 0.05*arma::trans(unit_vec(i-1,n_vals));
        }

        simplex_fn_vals(i) = opt_objfn(simplex_points.row(i).t(),nullptr,opt_data);

        if (vals_bound) {
            simplex_points.row(i) = arma::trans( transform(simplex_points.row(i).t(), bounds_type, lower_bounds, upper_bounds) );
        }
    }

    double min_val = simplex_fn_vals.min();

    //
    // begin loop

    if (verbose_print_level > 0)
    {
        std::cout << "\nNelder-Mead: beginning search...\n";

        if (verbose_print_level == 2)
        {
            std::cout << "  - Initialization Phase:\n";
            arma::cout << "    Objective function value at each vertex:\n" << simplex_fn_vals.t() << "\n";
            arma::cout << "    Simplex matrix:\n" << simplex_points << "\n";
        }
    }

    uint_t iter = 0;
    double err = 2*err_tol;

    while (err > err_tol && iter < iter_max)
    {
        iter++;
        bool next_iter = false;
        
        // step 1

        arma::uvec sort_vec = arma::sort_index(simplex_fn_vals); // sort from low (best) to high (worst) values

        simplex_fn_vals = simplex_fn_vals(sort_vec);
        simplex_points = simplex_points.rows(sort_vec);

        // step 2

        arma::vec centroid = arma::trans(arma::sum(simplex_points.rows(0,n_vals-1),0)) / static_cast<double>(n_vals);

        arma::vec x_r = centroid + par_alpha*(centroid - simplex_points.row(n_vals).t());

        double f_r = box_objfn(x_r,nullptr,opt_data);

        if (f_r >= simplex_fn_vals(0) && f_r < simplex_fn_vals(n_vals-1)) 
        {   // reflected point is neither best nor worst in the new simplex
            simplex_points.row(n_vals) = x_r.t();
            next_iter = true;
        }

        // step 3

        if (!next_iter && f_r < simplex_fn_vals(0)) 
        {   // reflected point is better than the current best; try to go farther along this direction
            arma::vec x_e = centroid + par_gamma*(x_r - centroid);

            double f_e = box_objfn(x_e,nullptr,opt_data);

            if (f_e < f_r) {
                simplex_points.row(n_vals) = x_e.t();
            } else {
                simplex_points.row(n_vals) = x_r.t();
            }

            next_iter = true;
        }

        // steps 4, 5, 6

        if (!next_iter && f_r >= simplex_fn_vals(n_vals-1)) 
        {   // reflected point is still worse than x_n; contract

            // steps 4 and 5

            if (f_r < simplex_fn_vals(n_vals)) 
            {   // outside contraction
                arma::vec x_oc = centroid + par_beta*(x_r - centroid);

                double f_oc = box_objfn(x_oc,nullptr,opt_data);

                if (f_oc <= f_r)
                {
                    simplex_points.row(n_vals) = x_oc.t();
                    next_iter = true;
                }
            } 
            else 
            {   // inside contraction: f_r >= simplex_fn_vals(n_vals)
                
                // x_ic = centroid - par_beta*(x_r - centroid);
                arma::vec x_ic = centroid + par_beta*(simplex_points.row(n_vals).t() - centroid);

                double f_ic = box_objfn(x_ic,nullptr,opt_data);

                if (f_ic < simplex_fn_vals(n_vals))
                {
                    simplex_points.row(n_vals) = x_ic.t();
                    next_iter = true;
                }
            }
        }

        // step 6

        if (!next_iter) 
        {   // neither outside nor inside contraction was acceptable; shrink the simplex toward x(0)
            for (size_t i=1; i < n_vals + 1; i++) {
                simplex_points.row(i) = simplex_points.row(0) + par_delta*(simplex_points.row(i) - simplex_points.row(0));
            }
        }

        // check change in fn_val
#ifdef OPTIM_USE_OMP
        #pragma omp parallel for
#endif
        for (size_t i=0; i < n_vals + 1; i++) {
            simplex_fn_vals(i) = box_objfn(simplex_points.row(i).t(),nullptr,opt_data);
        }

        //
    
        err = std::abs(min_val - simplex_fn_vals.max());
        min_val = simplex_fn_vals.min();

        // printing

        if (verbose_print_level > 0)
        {
            std::cout << "  - Iteration: " << iter << "\n";
            std::cout << "    min_val:   " << min_val << "\n";

            if (verbose_print_level == 1)
            {
                printf("\n");
                arma::cout << "    Current optimal input values:\n";
                arma::cout << simplex_points.row(index_min(simplex_fn_vals)) << "\n";
            }

            if (verbose_print_level == 2)
            {
                printf("\n");
                arma::cout << "    Objective function value at each vertex:\n" << simplex_fn_vals.t() << "\n";
                arma::cout << "    Simplex matrix:\n" << simplex_points << "\n";
            }
        }

    }

    if (verbose_print_level > 0) {
        std::cout << "Nelder-Mead: search completed.\n";
    }

    //

    arma::vec prop_out = simplex_points.row(index_min(simplex_fn_vals)).t();
    
    if (vals_bound) {
        prop_out = inv_transform(prop_out, bounds_type, lower_bounds, upper_bounds);
    }

    error_reporting(init_out_vals,prop_out,opt_objfn,opt_data,success,err,err_tol,iter,iter_max,conv_failure_switch,settings_inp);

    //
    
    return success;
}

inline
bool
nm(arma::vec& init_out_vals, std::function<double (const arma::vec& vals_inp, arma::vec* grad_out, void* opt_data)> opt_objfn, void* opt_data)
{
    return nm_int(init_out_vals,opt_objfn,opt_data,nullptr);
}

inline
bool
nm(arma::vec& init_out_vals, std::function<double (const arma::vec& vals_inp, arma::vec* grad_out, void* opt_data)> opt_objfn, void* opt_data, algo_settings_t& settings)
{
    return nm_int(init_out_vals,opt_objfn,opt_data,&settings);
}

#endif
