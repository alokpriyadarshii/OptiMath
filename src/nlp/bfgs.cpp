@@ -62,63 +62,79 @@ core::SolveResult<BFGSResult> minimize_bfgs(
        if (gx.norm2() <= grad_tol) {
            BFGSResult out;
            out.solution = {x, fx};
            out.stats = stats;

            core::SolveResult<BFGSResult> res;
            res.status = core::Status::Ok();
            res.solution = out;
            res.solve_time_seconds = timer.elapsed_seconds();
            res.iterations = it;
            return res;
        }

        // Search direction p = -H g
        optimath::linalg::Vector p = negate(matvec_local(H, gx));

        // Ensure descent; if not, reset.
        const double gtp = gx.dot(p);
        if (gtp >= -1e-16) {
            H = identity(n);
            p = negate(gx);
        }

        // Backtracking Armijo line search.
        double alpha = 1.0;
        double fnew = fx;
        bool accepted = false;
        optimath::linalg::Vector xnew = x;
        for (int ls = 0; ls < 60; ++ls) {
            xnew = x + alpha * p;
            fnew = f.value(xnew);
            stats.function_evals++;
            if (fnew <= fx + c1 * alpha * gtp) {
                fx = fnew;
                accepted = true;
                break;
            }
            alpha *= 0.5;
            if (alpha < 1e-16) break;
        }

        if (!accepted) {
            BFGSResult out;
            out.solution = {x, fx};
            out.stats = stats;

            core::SolveResult<BFGSResult> res;
            res.status = core::Status::Numerical("BFGS: line search failed");
            res.solution = out;
            res.solve_time_seconds = timer.elapsed_seconds();
            res.iterations = it;
            return res;
        }

        optimath::linalg::Vector gnew = eval_grad(xnew);
        stats.gradient_evals++;

        optimath::linalg::Vector s = xnew - x;
        optimath::linalg::Vector y = gnew - gx;

        const double ys = y.dot(s);
        if (std::abs(ys) < 1e-18) {
            H = identity(n);
        } else {
            const double rho = 1.0 / ys;

            // BFGS update: H = (I - rho s y^T) H (I - rho y s^T) + rho s s^T
            // Implement with explicit loops (small/medium n).
            optimath::linalg::Matrix I = identity(n);
            optimath::linalg::Matrix A(n, n, 0.0);
            optimath::linalg::Matrix B(n, n, 0.0);
            for (std::size_t i = 0; i < n; ++i) {
                for (std::size_t j = 0; j < n; ++j) {
                    A(i, j) = I(i, j) - rho * s.raw()[i] * y.raw()[j];
                    B(i, j) = I(i, j) - rho * y.raw()[i] * s.raw()[j];
                }
            }

            // tmp = A * H
