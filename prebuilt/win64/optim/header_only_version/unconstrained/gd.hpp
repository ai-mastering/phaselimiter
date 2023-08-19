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
 * Gradient Descent (GD)
 */

#ifndef _optim_gd_HPP
#define _optim_gd_HPP

bool gd_basic_int(arma::vec& init_out_vals, std::function<double (const arma::vec& vals_inp, arma::vec* grad_out, void* opt_data)> opt_objfn, void* opt_data, algo_settings_t* settings_inp);

bool gd(arma::vec& init_out_vals, std::function<double (const arma::vec& vals_inp, arma::vec* grad_out, void* opt_data)> opt_objfn, void* opt_data);
bool gd(arma::vec& init_out_vals, std::function<double (const arma::vec& vals_inp, arma::vec* grad_out, void* opt_data)> opt_objfn, void* opt_data, algo_settings_t& settings);

// internal update function

inline
arma::vec
gd_update(const arma::vec& vals_inp, const arma::vec& grad, const arma::vec& grad_p, const arma::vec& direc,
          std::function<double (const arma::vec& vals_inp, arma::vec* grad_out, void* opt_data)> box_objfn, void* opt_data,
          const uint_t iter, const uint_t gd_method, gd_settings_t& gd_settings,
          arma::vec& adam_vec_m, arma::vec& adam_vec_v)
{
    arma::vec direc_out; // direction

    if (gd_settings.step_decay)
    {
        if (iter % gd_settings.step_decay_periods == 0)
        {
            gd_settings.step_size *= gd_settings.step_decay_val;
        }
    }

    switch (gd_method)
    {
        case 0: // basic
        {
            direc_out = gd_settings.step_size * grad_p;
            break;
        }

        case 1: // momentum
        {
            // direc_out = gd_settings.step_size * (gd_settings.momentum_par * direc + grad_p);
            direc_out = gd_settings.momentum_par * direc + gd_settings.step_size * grad_p;
            break;
        }

        case 2: // Nesterov accelerated gradient
        {
            arma::vec NAG_grad(vals_inp.n_elem);
            box_objfn(vals_inp - gd_settings.momentum_par * direc, &NAG_grad, opt_data);

            // direc_out = gd_settings.step_size * (gd_settings.momentum_par * direc + NAG_grad);
            direc_out = gd_settings.momentum_par * direc + gd_settings.step_size * NAG_grad;
            break;
        }

        case 3: // AdaGrad
        {
            adam_vec_v += arma::pow(grad_p,2);

            direc_out = gd_settings.step_size * grad_p / (arma::sqrt(adam_vec_v) + gd_settings.norm_term);
            break;
        }

        case 4: // RMSProp
        {
            adam_vec_v = gd_settings.ada_rho * adam_vec_v + (1.0 - gd_settings.ada_rho) * arma::pow(grad_p,2);

            direc_out = gd_settings.step_size * grad_p / (arma::sqrt(adam_vec_v) + gd_settings.norm_term);
            break;
        }

        case 5: // Adadelta
        {
            if (iter == 1) {
                adam_vec_m += gd_settings.step_size;
            }
            adam_vec_v = gd_settings.ada_rho * adam_vec_v + (1.0 - gd_settings.ada_rho) * arma::pow(grad_p,2);

            direc_out = grad_p % (arma::sqrt(adam_vec_m) + gd_settings.norm_term) / (arma::sqrt(adam_vec_v) + gd_settings.norm_term);

            adam_vec_m = gd_settings.ada_rho * adam_vec_m + (1.0 - gd_settings.ada_rho) * arma::pow(direc_out,2);
            break;
        }

        case 6: // Adam and AdaMax
        {
            adam_vec_m = gd_settings.adam_beta_1 * adam_vec_m + (1.0 - gd_settings.adam_beta_1) * grad_p;

            if (gd_settings.ada_max)
            {
                adam_vec_v = arma::max(gd_settings.adam_beta_2 * adam_vec_v, arma::abs(grad_p));

                double adam_step_size = gd_settings.step_size / (1.0 - std::pow(gd_settings.adam_beta_1,iter));

                direc_out = adam_step_size * adam_vec_m / (adam_vec_v + gd_settings.norm_term);
            }
            else
            {
                double adam_step_size = gd_settings.step_size * std::sqrt(1.0 - std::pow(gd_settings.adam_beta_2,iter)) \
                                     / (1.0 - std::pow(gd_settings.adam_beta_1,iter));

                adam_vec_v = gd_settings.adam_beta_2 * adam_vec_v + (1.0 - gd_settings.adam_beta_2) * arma::pow(grad_p,2);

                direc_out = adam_step_size * adam_vec_m / (arma::sqrt(adam_vec_v) + gd_settings.norm_term);
            }

            break;
        }

        case 7: // Nadam and NadaMax
        {
            adam_vec_m = gd_settings.adam_beta_1 * adam_vec_m + (1.0 - gd_settings.adam_beta_1) * grad_p;

            if (gd_settings.ada_max)
            {
                adam_vec_v = arma::max(gd_settings.adam_beta_2 * adam_vec_v, arma::abs(grad_p));

                arma::vec m_hat = adam_vec_m / (1.0 - std::pow(gd_settings.adam_beta_1,iter));
                arma::vec grad_hat = grad_p / (1.0 - std::pow(gd_settings.adam_beta_1,iter));

                direc_out = gd_settings.step_size * ( gd_settings.adam_beta_1 * m_hat + (1.0 - gd_settings.adam_beta_1) * grad_hat ) \
                            / (adam_vec_v + gd_settings.norm_term);
            }
            else
            {
                adam_vec_v = gd_settings.adam_beta_2 * adam_vec_v + (1.0 - gd_settings.adam_beta_2) * arma::pow(grad_p,2);

                arma::vec m_hat = adam_vec_m / (1.0 - std::pow(gd_settings.adam_beta_1,iter));
                arma::vec v_hat = adam_vec_v / (1.0 - std::pow(gd_settings.adam_beta_2,iter));
                arma::vec grad_hat = grad_p / (1.0 - std::pow(gd_settings.adam_beta_1,iter));

                direc_out = gd_settings.step_size * ( gd_settings.adam_beta_1 * m_hat + (1.0 - gd_settings.adam_beta_1) * grad_hat ) \
                            / (arma::sqrt(v_hat) + gd_settings.norm_term);
            }

            break;
        }

        default:
        {
            printf("error: unknown value for gd_method");
            break;
        }
    }

    return direc_out;
}

//

inline
bool
gd_basic_int(arma::vec& init_out_vals, std::function<double (const arma::vec& vals_inp, arma::vec* grad_out, void* opt_data)> opt_objfn, void* opt_data, algo_settings_t* settings_inp)
{
    // notation: 'p' stands for '+1'.

    bool success = false;
    
    const size_t n_vals = init_out_vals.n_elem;

    //
    // GD settings

    algo_settings_t settings;

    if (settings_inp) {
        settings = *settings_inp;
    }
    
    const uint_t conv_failure_switch = settings.conv_failure_switch;
    const uint_t iter_max = settings.iter_max;
    const double err_tol = settings.err_tol;

    gd_settings_t gd_settings = settings.gd_settings;

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
            
            double ret;
            
            if (grad_out) {
                arma::vec grad_obj = *grad_out;

                ret = opt_objfn(vals_inv_trans,&grad_obj,opt_data);

                // arma::mat jacob_matrix = jacobian_adjust(vals_inp,bounds_type,lower_bounds,upper_bounds);
                arma::vec jacob_vec = arma::diagvec(jacobian_adjust(vals_inp,bounds_type,lower_bounds,upper_bounds));

                // *grad_out = jacob_matrix * grad_obj; //
                *grad_out = jacob_vec % grad_obj; //
            }
            else
            {
                ret = opt_objfn(vals_inv_trans,nullptr,opt_data);
            }

            return ret;
        }
        else
        {
            return opt_objfn(vals_inp,grad_out,opt_data);
        }
    };

    //
    // initialization

    arma::vec x = init_out_vals;

    if (!x.is_finite())
    {
        printf("gd error: non-finite initial value(s).\n");
        return false;
    }

    if (vals_bound) { // should we transform the parameters?
        x = transform(x, bounds_type, lower_bounds, upper_bounds);
    }

    arma::vec grad(n_vals); // gradient
    box_objfn(x,&grad,opt_data);

    double err = arma::norm(grad, 2);
    if (err <= err_tol) {
        return true;
    }

    //

    arma::vec d = grad, d_p;
    arma::vec x_p = x, grad_p = grad;

    err = arma::norm(grad_p, 2);
    if (err <= err_tol)
    {
        init_out_vals = x_p;
        return true;
    }

    //

    arma::vec adam_vec_m;
    arma::vec adam_vec_v;

    if (settings.gd_method == 3 || settings.gd_method == 4)
    {
        adam_vec_v = arma::zeros(n_vals);
    }

    if (settings.gd_method == 5 || settings.gd_method == 6 || settings.gd_method == 7)
    {
        adam_vec_m = arma::zeros(n_vals);
        adam_vec_v = arma::zeros(n_vals);
    }

    //
    // begin loop

    uint_t iter = 0;

    while (err > err_tol && iter < iter_max)
    {
        iter++;

        //

        d_p = gd_update(x,grad,grad_p,d,box_objfn,opt_data,iter,
                        settings.gd_method,gd_settings,adam_vec_m,adam_vec_v);

        x_p = x - d_p;
        grad = grad_p;

        box_objfn(x_p,&grad_p,opt_data);

        //

        err = arma::norm(grad_p, 2);

        d = d_p;
        x = x_p;
    }

    //

    if (vals_bound)
    {
        x_p = inv_transform(x_p, bounds_type, lower_bounds, upper_bounds);
    }

    error_reporting(init_out_vals,x_p,opt_objfn,opt_data,success,err,err_tol,iter,iter_max,conv_failure_switch,settings_inp);

    //

    return success;
}

inline
bool
gd(arma::vec& init_out_vals, std::function<double (const arma::vec& vals_inp, arma::vec* grad_out, void* opt_data)> opt_objfn, void* opt_data)
{
    return gd_basic_int(init_out_vals,opt_objfn,opt_data,nullptr);
}

inline
bool
gd(arma::vec& init_out_vals, std::function<double (const arma::vec& vals_inp, arma::vec* grad_out, void* opt_data)> opt_objfn, void* opt_data, algo_settings_t& settings)
{
    return gd_basic_int(init_out_vals,opt_objfn,opt_data,&settings);
}

#endif
