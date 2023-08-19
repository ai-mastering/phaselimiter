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
 * Particle Swarm Optimization (PSO)
 */

#ifndef _optim_pso_HPP
#define _optim_pso_HPP

bool pso_int(arma::vec& init_out_vals, std::function<double (const arma::vec& vals_inp, arma::vec* grad_out, void* opt_data)> opt_objfn, void* opt_data, algo_settings_t* settings_inp);

bool pso(arma::vec& init_out_vals, std::function<double (const arma::vec& vals_inp, arma::vec* grad_out, void* opt_data)> opt_objfn, void* opt_data);
bool pso(arma::vec& init_out_vals, std::function<double (const arma::vec& vals_inp, arma::vec* grad_out, void* opt_data)> opt_objfn, void* opt_data, algo_settings_t& settings);

//

inline
bool
pso_int(arma::vec& init_out_vals, std::function<double (const arma::vec& vals_inp, arma::vec* grad_out, void* opt_data)> opt_objfn, void* opt_data, algo_settings_t* settings_inp)
{
    bool success = false;

    const size_t n_vals = init_out_vals.n_elem;

    //
    // PSO settings

    algo_settings_t settings;

    if (settings_inp) {
        settings = *settings_inp;
    }

    const uint_t conv_failure_switch = settings.conv_failure_switch;
    const double err_tol = settings.err_tol;

    const bool center_particle = settings.pso_center_particle;

    const size_t n_pop = (center_particle) ? settings.pso_n_pop + 1 : settings.pso_n_pop;
    const size_t n_gen = settings.pso_n_gen;
    const uint_t check_freq = (settings.pso_check_freq > 0) ? settings.pso_check_freq : n_gen ;

    const uint_t inertia_method = settings.pso_inertia_method;

    double par_w = settings.pso_par_initial_w;
    const double par_w_max = settings.pso_par_w_max;
    const double par_w_min = settings.pso_par_w_min;
    const double par_damp = settings.pso_par_w_damp;

    const uint_t velocity_method = settings.pso_velocity_method;

    double par_c_cog = settings.pso_par_c_cog;
    double par_c_soc = settings.pso_par_c_soc;

    const double par_initial_c_cog = settings.pso_par_initial_c_cog;
    const double par_final_c_cog = settings.pso_par_final_c_cog;
    const double par_initial_c_soc = settings.pso_par_initial_c_soc;
    const double par_final_c_soc = settings.pso_par_final_c_soc;

    const arma::vec par_initial_lb = (settings.pso_initial_lb.n_elem == n_vals) ? settings.pso_initial_lb : init_out_vals - 0.5;
    const arma::vec par_initial_ub = (settings.pso_initial_ub.n_elem == n_vals) ? settings.pso_initial_ub : init_out_vals + 0.5;

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
    // initialize

    arma::vec objfn_vals(n_pop);
    arma::mat P(n_pop,n_vals);

#ifdef OPTIM_USE_OMP
    #pragma omp parallel for
#endif
    for (size_t i=0; i < n_pop; i++) 
    {
        if (center_particle && i == n_pop - 1) {
            P.row(i) = arma::sum(P.rows(0,n_pop-2),0) / static_cast<double>(n_pop-1); // center vector
        } else {
            P.row(i) = par_initial_lb.t() + (par_initial_ub.t() - par_initial_lb.t())%arma::randu(1,n_vals);
        }

        double prop_objfn_val = opt_objfn(P.row(i).t(),nullptr,opt_data);

        if (!std::isfinite(prop_objfn_val)) {
            prop_objfn_val = inf;
        }
        
        objfn_vals(i) = prop_objfn_val;

        if (vals_bound) {
            P.row(i) = arma::trans( transform(P.row(i).t(), bounds_type, lower_bounds, upper_bounds) );
        }
    }

    arma::vec best_vals = objfn_vals;
    arma::mat best_vecs = P;

    double global_best_val = objfn_vals.min();
    double global_best_val_check = global_best_val;
    arma::rowvec global_best_vec = P.row( objfn_vals.index_min() );

    //
    // begin loop

    uint_t iter = 0;
    double err = 2.0*err_tol;

    arma::mat V = arma::zeros(n_pop,n_vals);

    while (err > err_tol && iter < n_gen)
    {
        iter++;
        
        //
        // parameter updating

        if (inertia_method == 1) {
            par_w = par_w_min + (par_w_max - par_w_min) * (iter + 1) / n_gen;
        } else {
            par_w *= par_damp;
        }

        if (velocity_method == 2)
        {
            par_c_cog = par_initial_c_cog - (par_initial_c_cog - par_final_c_cog) * (iter + 1) / n_gen;
            par_c_soc = par_initial_c_soc - (par_initial_c_soc - par_final_c_soc) * (iter + 1) / n_gen;
        }

        //
        // population loop

#ifdef OPTIM_USE_OMP
        #pragma omp parallel for 
#endif
        for (size_t i=0; i < n_pop; i++)
        {
            if ( !(center_particle && i == n_pop - 1) )
            {
                V.row(i) = par_w*V.row(i) + par_c_cog*arma::randu(1,n_vals)%(best_vecs.row(i) - P.row(i)) + par_c_soc*arma::randu(1,n_vals)%(global_best_vec - P.row(i));

                P.row(i) += V.row(i);
            }
            else
            {
                P.row(i) = arma::sum(P.rows(0,n_pop-2),0) / static_cast<double>(n_pop-1); // center vector
            }
            
            //

            double prop_objfn_val = box_objfn(P.row(i).t(),nullptr,opt_data);

            if (!std::isfinite(prop_objfn_val)) {
                prop_objfn_val = inf;
            }
        
            objfn_vals(i) = prop_objfn_val;
                
            if (objfn_vals(i) < best_vals(i))
            {
                best_vals(i) = objfn_vals(i);
                best_vecs.row(i) = P.row(i);
            }
        }

        uint_t min_objfn_val_index = best_vals.index_min();
        double min_objfn_val = best_vals(min_objfn_val_index);

        if (min_objfn_val < global_best_val)
        {
            global_best_val = min_objfn_val;
            global_best_vec = best_vecs.row( min_objfn_val_index );
        }

        if (iter%check_freq == 0) 
        {   
            err = std::abs(global_best_val - global_best_val_check) / (1.0 + std::abs(global_best_val));
            
            if (global_best_val < global_best_val_check) {
                global_best_val_check = global_best_val;
            }
        }
    }

    //

    if (vals_bound) {
        global_best_vec = arma::trans( inv_transform(global_best_vec.t(), bounds_type, lower_bounds, upper_bounds) );
    }

    error_reporting(init_out_vals,global_best_vec.t(),opt_objfn,opt_data,success,err,err_tol,iter,n_gen,conv_failure_switch,settings_inp);

    //

    return true;
}

inline
bool
pso(arma::vec& init_out_vals, std::function<double (const arma::vec& vals_inp, arma::vec* grad_out, void* opt_data)> opt_objfn, void* opt_data)
{
    return pso_int(init_out_vals,opt_objfn,opt_data,nullptr);
}

inline
bool
pso(arma::vec& init_out_vals, std::function<double (const arma::vec& vals_inp, arma::vec* grad_out, void* opt_data)> opt_objfn, void* opt_data, algo_settings_t& settings)
{
    return pso_int(init_out_vals,opt_objfn,opt_data,&settings);
}

#endif
