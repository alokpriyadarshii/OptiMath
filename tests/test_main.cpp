#include <iostream>
#include <vector>

// Each translation unit defines one of these.
void test_lp_simplex();
void test_nlp_bfgs();

int main() {
    struct Test { const char* name; void (*fn)(); };
    std::vector<Test> tests = {
        {"lp_simplex", &test_lp_simplex},
        {"nlp_bfgs", &test_nlp_bfgs},
    };

    int passed = 0;
    for (const auto& t : tests) {
        try {
            t.fn();
            std::cout << "[PASS] " << t.name << "\n";
            ++passed;
        } catch (const std::exception& e) {
            std::cout << "[FAIL] " << t.name << "\n";
            std::cout << "       " << e.what() << "\n";
        } catch (...) {
            std::cout << "[FAIL] " << t.name << "\n";
            std::cout << "       unknown exception\n";
        }
    }

    std::cout << passed << "/" << static_cast<int>(tests.size()) << " tests passed\n";
    return (passed == static_cast<int>(tests.size())) ? 0 : 1;
}
