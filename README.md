# OptiMath

set -euo pipefail

# 1) Go to project folder (adjust if you're already there)
cd "OptiMath"

# 2) Ensure build tools exist (macOS, C++ only)
xcode-select -p >/dev/null 2>&1 || xcode-select --install || true
command -v brew >/dev/null 2>&1 || { echo "Homebrew is required. Install it, then rerun."; exit 1; }
command -v cmake >/dev/null 2>&1 || brew install cmake
command -v ninja >/dev/null 2>&1 || brew install ninja

# 3) Configure + build (Release)
rm -rf build
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DOPTIMATH_BUILD_APPS=ON -DOPTIMATH_BUILD_TESTS=ON
cmake --build build -j "$(sysctl -n hw.ncpu)"

# 4) Run tests
ctest --test-dir build --output-on-failure

# 5) Verify CLI exists + show help
echo "== cli help =="
./build/optimath_cli --help

# 6) List demos
echo "== demos =="
./build/optimath_cli --list

# 7) Run LP demos
echo "== lp:diet =="
./build/optimath_cli lp:diet

echo "== lp:transport =="
./build/optimath_cli lp:transport

# 8) Run MIP demo
echo "== mip:knapsack =="
./build/optimath_cli mip:knapsack

# 9) Run NLP demos
echo "== nlp:rosenbrock =="
./build/optimath_cli nlp:rosenbrock

echo "== nlp:curvefit (tuned for OK) =="
./build/optimath_cli nlp:curvefit --max-it 200000 --primal-tol 1e-6

echo "== nlp:portfolio =="
./build/optimath_cli nlp:portfolio

echo "Done. Build dir: ./build"

