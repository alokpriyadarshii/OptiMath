#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

#include "optimath/core/options.hpp"
#include "optimath/core/status.hpp"
#include "optimath/linalg/vector.hpp"
#include "optimath/lp/linear_program.hpp"
#include "optimath/lp/mip_branch_and_bound.hpp"
#include "optimath/lp/simplex.hpp"
#include "optimath/nlp/bfgs.hpp"
#include "optimath/nlp/objective.hpp"
#include "optimath/nlp/penalty.hpp"

namespace {

using optimath::core::SolverOptions;
using optimath::lp::ConstraintSense;

void print_usage() {
    std::cout
        << "OptiMathCPP CLI\n\n"
        << "Usage:\n"
        << "  optimath_cli --list\n"
        << "  optimath_cli <demo> [--verbose] [--max-it N] [--time-limit S] [--primal-tol X] [--dual-tol X]\n\n"
        << "Demos:\n"
        << "  lp:diet         - diet style LP (min-cost via max(-cost))\n"
        << "  lp:transport    - transportation LP (min-cost via max(-cost))\n"
        << "  mip:knapsack    - 0/1 knapsack using branch-and-bound\n"
        << "  nlp:rosenbrock  - Rosenbrock minimization (BFGS)\n"
        << "  nlp:curvefit    - exponential curve fitting (BFGS)\n"
        << "  nlp:portfolio   - constrained portfolio (penalty + BFGS)\n\n";
}

bool is_flag(const std::string& s, const std::string& f) { return s == f; }

std::optional<double> parse_double(const std::string& s) {
    char* end = nullptr;
    const double v = std::strtod(s.c_str(), &end);
    if (end == s.c_str() || *end != '\0') return std::nullopt;
    return v;
}

std::optional<std::size_t> parse_size(const std::string& s) {
    char* end = nullptr;
    const unsigned long long v = std::strtoull(s.c_str(), &end, 10);
    if (end == s.c_str() || *end != '\0') return std::nullopt;
    return static_cast<std::size_t>(v);
}

SolverOptions parse_options(int argc, char** argv, int start_index) {
    SolverOptions opt;
    for (int i = start_index; i < argc; ++i) {
        const std::string a = argv[i];
        if (is_flag(a, "--verbose")) {
            opt.verbose = true;
        } else if (is_flag(a, "--max-it") && i + 1 < argc) {
            auto v = parse_size(argv[++i]);
            if (v) opt.limits.max_iterations = *v;
        } else if (is_flag(a, "--time-limit") && i + 1 < argc) {
            auto v = parse_double(argv[++i]);
            if (v) opt.limits.time_limit_seconds = *v;
        } else if (is_flag(a, "--primal-tol") && i + 1 < argc) {
            auto v = parse_double(argv[++i]);
            if (v) opt.tol.primal_feas = *v;
        } else if (is_flag(a, "--dual-tol") && i + 1 < argc) {
            auto v = parse_double(argv[++i]);
            if (v) opt.tol.dual_feas = *v;
        }
    }
    return opt;
}

void print_status(const optimath::core::Status& st) {
    std::cout << "Status: " << (st.ok() ? "OK" : "NOT OK")
              << (st.message.empty() ? "" : (" - " + st.message)) << "\n";
}

void print_vector(const std::vector<double>& x, const std::vector<std::string>& names = {}) {
    std::cout << std::fixed << std::setprecision(6);
    for (std::size_t i = 0; i < x.size(); ++i) {
        const std::string label = (i < names.size() && !names[i].empty()) ? names[i] : ("x" + std::to_string(i));
        std::cout << "  " << label << " = " << x[i] << "\n";
    }
}

void demo_lp_diet(const SolverOptions& opt) {
    // Minimize cost subject to nutrition >= requirements.
    // We solve by maximizing -cost.
    // Variables: [oats, milk, eggs]
    const std::vector<std::string> names = {"oats", "milk", "eggs"};

    optimath::lp::LinearProgram lp(3);
    lp.set_objective({-2.0, -1.0, -0.5}); // maximize -cost

    // Protein: 10*oats + 8*milk + 6*eggs >= 50
    lp.add_constraint({10.0, 8.0, 6.0}, 50.0, ConstraintSense::kGreaterEqual, "protein");

    // Calories: 300*oats + 150*milk + 70*eggs >= 1500
    lp.add_constraint({300.0, 150.0, 70.0}, 1500.0, ConstraintSense::kGreaterEqual, "calories");

    auto res = optimath::lp::solve_simplex(lp, opt);
    print_status(res.status);
    if (!res.status.ok()) return;

    const double min_cost = -res.solution.solution.objective_value;
    std::cout << "Min cost (objective): " << std::fixed << std::setprecision(6) << min_cost << "\n";
    print_vector(res.solution.solution.x, names);
    std::cout << "Pivots: " << res.solution.stats.pivots << "\n";
    std::cout << "Solve time: " << res.solve_time_seconds << "s\n";
}

void demo_lp_transport(const SolverOptions& opt) {
    // Two factories -> three warehouses (min cost).
    // Variables: x_ij flattened (i*W + j)
    // We'll minimize cost by maximizing -cost.
    const int F = 2, W = 3;

    const std::vector<double> supply = {35.0, 50.0};
    const std::vector<double> demand = {30.0, 30.0, 25.0};

    const std::vector<double> cost = {
        2.0, 4.0, 5.0, // factory0 -> w0,w1,w2
        3.0, 1.0, 7.0  // factory1 -> w0,w1,w2
    };

    optimath::lp::LinearProgram lp(static_cast<std::size_t>(F * W));

    std::vector<double> obj(F * W);
    for (int k = 0; k < F * W; ++k) obj[k] = -cost[k];
    lp.set_objective(obj);

    // Supply constraints: sum_j x_ij <= supply_i
    for (int i = 0; i < F; ++i) {
        std::vector<double> a(F * W, 0.0);
        for (int j = 0; j < W; ++j) a[i * W + j] = 1.0;
        lp.add_constraint(a, supply[i], ConstraintSense::kLessEqual, "supply_" + std::to_string(i));
    }

    // Demand constraints: sum_i x_ij >= demand_j
    for (int j = 0; j < W; ++j) {
        std::vector<double> a(F * W, 0.0);
        for (int i = 0; i < F; ++i) a[i * W + j] = 1.0;
        lp.add_constraint(a, demand[j], ConstraintSense::kGreaterEqual, "demand_" + std::to_string(j));
    }

    auto res = optimath::lp::solve_simplex(lp, opt);
    print_status(res.status);
    if (!res.status.ok()) return;

    const double min_cost = -res.solution.solution.objective_value;
    std::cout << "Min transport cost: " << std::fixed << std::setprecision(6) << min_cost << "\n";

    const auto& x = res.solution.solution.x;
    for (int i = 0; i < F; ++i) {
        for (int j = 0; j < W; ++j) {
            std::cout << "  x(" << i << "," << j << ") = " << std::fixed << std::setprecision(6) << x[i * W + j] << "\n";
        }
    }

    std::cout << "Pivots: " << res.solution.stats.pivots << "\n";
    std::cout << "Solve time: " << res.solve_time_seconds << "s\n";
}

void demo_mip_knapsack(const SolverOptions& opt) {
    // 0/1 knapsack as MIP:
    // maximize v^T x
    // s.t. w^T x <= W
    // x_i in {0,1}
    const std::vector<double> value = {6, 10, 12, 7};
    const std::vector<double> weight = {2, 4, 6, 3};
    const double capacity = 9;
    const std::size_t n = value.size();

    optimath::lp::LinearProgram lp(n);
    lp.set_objective(value);
    lp.add_constraint(weight, capacity, ConstraintSense::kLessEqual, "capacity");

    optimath::lp::MIPModel mip;
    mip.relaxation = lp;
    mip.integer_vars = {0, 1, 2, 3};

    auto res = optimath::lp::solve_branch_and_bound(mip, opt);
    print_status(res.status);
    if (!res.status.ok()) return;

    if (!res.solution.has_feasible_solution) {
        std::cout << "No feasible integer solution found within limits.\n";
        return;
    }

    std::cout << "Best integer objective: " << std::fixed << std::setprecision(6) << res.solution.solution.objective_value << "\n";
    print_vector(res.solution.solution.x);
    std::cout << "Nodes explored: " << res.solution.stats.nodes_explored << "\n";
    std::cout << "LP solves: " << res.solution.stats.lp_solves << "\n";
    std::cout << "Solve time: " << res.solve_time_seconds << "s\n";
}

void demo_nlp_rosenbrock(const SolverOptions& opt) {
    // Rosenbrock: f(x,y) = (1-x)^2 + 100(y - x^2)^2
    using optimath::linalg::Vector;
    optimath::nlp::LambdaObjective f(
        [](const Vector& x) {
            const double a = 1.0 - x.raw()[0];
            const double b = x.raw()[1] - x.raw()[0] * x.raw()[0];
            return a * a + 100.0 * b * b;
        },
        [](const Vector& x) {
            const double dx = -2.0 * (1.0 - x.raw()[0]) - 400.0 * x.raw()[0] * (x.raw()[1] - x.raw()[0] * x.raw()[0]);
            const double dy = 200.0 * (x.raw()[1] - x.raw()[0] * x.raw()[0]);
            return Vector{dx, dy};
        });

    Vector x0{-1.2, 1.0};
    auto res = optimath::nlp::minimize_bfgs(f, x0, opt);

    print_status(res.status);
    std::cout << "Objective: " << std::fixed << std::setprecision(12) << res.solution.solution.objective_value << "\n";
    std::cout << "x: [" << res.solution.solution.x.raw()[0] << ", " << res.solution.solution.x.raw()[1] << "]\n";
    std::cout << "Iterations: " << res.solution.stats.iterations << "\n";
    std::cout << "f evals: " << res.solution.stats.function_evals << ", g evals: " << res.solution.stats.gradient_evals << "\n";
    std::cout << "Solve time: " << res.solve_time_seconds << "s\n";
}

void demo_nlp_curvefit(const SolverOptions& opt) {
    // Fit y(t) = a * exp(b t)
    // Objective: sum_i (a*exp(b t_i) - y_i)^2
    using optimath::linalg::Vector;

    const std::vector<double> t = {0.0, 0.5, 1.0, 1.5, 2.0};
    const std::vector<double> y = {1.02, 1.66, 2.73, 4.40, 7.39};

    optimath::nlp::LambdaObjective f(
        [&](const Vector& x) {
            const double a = x.raw()[0];
            const double b = x.raw()[1];
            double sse = 0.0;
            for (std::size_t i = 0; i < t.size(); ++i) {
                const double pred = a * std::exp(b * t[i]);
                const double r = pred - y[i];
                sse += r * r;
            }
            return sse;
        },
        [&](const Vector& x) {
            const double a = x.raw()[0];
            const double b = x.raw()[1];
            double ga = 0.0;
            double gb = 0.0;
            for (std::size_t i = 0; i < t.size(); ++i) {
                const double e = std::exp(b * t[i]);
                const double pred = a * e;
                const double r = pred - y[i];
                ga += 2.0 * r * e;
                gb += 2.0 * r * (a * e * t[i]);
            }
            return Vector{ga, gb};
        });

    Vector x0{1.0, 1.0};
    auto res = optimath::nlp::minimize_bfgs(f, x0, opt);

    print_status(res.status);
    std::cout << "Best-fit params: a=" << std::fixed << std::setprecision(8) << res.solution.solution.x.raw()[0]
              << ", b=" << res.solution.solution.x.raw()[1] << "\n";
    std::cout << "SSE: " << std::fixed << std::setprecision(12) << res.solution.solution.objective_value << "\n";
}

void demo_nlp_portfolio(const SolverOptions& opt) {
    // Mean-variance with constraints:
    // minimize 0.5 x^T Sigma x - mu^T x
    // s.t. sum x = 1, x >= 0
    using optimath::linalg::Vector;

    const std::vector<std::vector<double>> Sigma = {
        {0.10, 0.02, 0.04},
        {0.02, 0.08, 0.01},
        {0.04, 0.01, 0.12}
    };
    const std::vector<double> mu = {0.12, 0.10, 0.14};

    optimath::nlp::LambdaObjective f(
        [&](const Vector& x) {
            double quad = 0.0;
            for (std::size_t i = 0; i < 3; ++i) {
                for (std::size_t j = 0; j < 3; ++j) quad += x.raw()[i] * Sigma[i][j] * x.raw()[j];
            }
            double lin = 0.0;
            for (std::size_t i = 0; i < 3; ++i) lin += mu[i] * x.raw()[i];
            return 0.5 * quad - lin;
        },
        [&](const Vector& x) {
            // grad = Sigma x - mu
            std::vector<double> g(3, 0.0);
            for (std::size_t i = 0; i < 3; ++i) {
                for (std::size_t j = 0; j < 3; ++j) g[i] += Sigma[i][j] * x.raw()[j];
                g[i] -= mu[i];
            }
            return Vector{g[0], g[1], g[2]};
        });

    std::vector<optimath::nlp::InequalityConstraint> ineq;
    for (std::size_t i = 0; i < 3; ++i) {
        ineq.push_back({
            [i](const Vector& x) { return -x.raw()[i]; },
            [i](const Vector& /*x*/) {
                std::vector<double> g(3, 0.0);
                g[i] = -1.0;
                return Vector{g[0], g[1], g[2]};
            }
        });
    }

    std::vector<optimath::nlp::EqualityConstraint> eq;
    eq.push_back({
        [](const Vector& x) { return x.raw()[0] + x.raw()[1] + x.raw()[2] - 1.0; },
        [](const Vector& /*x*/) { return Vector{1.0, 1.0, 1.0}; }
    });

    Vector x0{1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0};
    auto res = optimath::nlp::minimize_with_penalty(f, x0, ineq, eq, opt);

    print_status(res.status);
    std::cout << "Objective: " << std::fixed << std::setprecision(12) << res.solution.solution.objective_value << "\n";
    std::cout << "Weights:\n";
    const auto& x = res.solution.solution.x.raw();
    std::cout << "  w0=" << x[0] << "\n  w1=" << x[1] << "\n  w2=" << x[2] << "\n";
}

} // namespace

int main(int argc, char** argv) {
    if (argc <= 1) {
        print_usage();
        return 0;
    }

    const std::string cmd = argv[1];
    if (cmd == "--help" || cmd == "-h") {
        print_usage();
        return 0;
    }

    if (cmd == "--list") {
        std::cout << "lp:diet\nlp:transport\nmip:knapsack\nnlp:rosenbrock\nnlp:curvefit\nnlp:portfolio\n";
        return 0;
    }

    const SolverOptions opt = parse_options(argc, argv, 2);

    if (cmd == "lp:diet") {
        demo_lp_diet(opt);
        return 0;
    }
    if (cmd == "lp:transport") {
        demo_lp_transport(opt);
        return 0;
    }
    if (cmd == "mip:knapsack") {
        demo_mip_knapsack(opt);
        return 0;
    }
    if (cmd == "nlp:rosenbrock") {
        demo_nlp_rosenbrock(opt);
        return 0;
    }
    if (cmd == "nlp:curvefit") {
        demo_nlp_curvefit(opt);
        return 0;
    }
    if (cmd == "nlp:portfolio") {
        demo_nlp_portfolio(opt);
        return 0;
    }

    std::cerr << "Unknown demo: " << cmd << "\n\n";
    print_usage();
    return 2;
}
