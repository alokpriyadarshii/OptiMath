const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => Array.from(document.querySelectorAll(selector));

const solverSelect = $('#solverSelect');
const solverForm = $('#solverForm');
const output = $('#output');
const outputTitle = $('#outputTitle');
const copyOutput = $('#copyOutput');

function parseNumbers(raw, label) {
  const values = raw
    .split(/[\n,]+/)
    .map((part) => part.trim())
    .filter(Boolean)
    .map(Number);

  if (!values.length || values.some((value) => !Number.isFinite(value))) {
    throw new Error(`${label} must contain valid numbers separated by commas.`);
  }

  return values;
}

function formatNumber(value) {
  if (!Number.isFinite(value)) return String(value);
  if (Math.abs(value) < 1e-10) return '0.000000';
  return value.toFixed(6);
}

function dot(a, b) {
  return a.reduce((sum, value, index) => sum + value * b[index], 0);
}

function norm2(values) {
  return Math.sqrt(values.reduce((sum, value) => sum + value * value, 0));
}

function solveLinearSystem(A, b) {
  const n = A.length;
  const M = A.map((row, i) => [...row, b[i]]);

  for (let col = 0; col < n; col += 1) {
    let pivot = col;
    for (let row = col + 1; row < n; row += 1) {
      if (Math.abs(M[row][col]) > Math.abs(M[pivot][col])) pivot = row;
    }

    if (Math.abs(M[pivot][col]) < 1e-10) return null;
    [M[col], M[pivot]] = [M[pivot], M[col]];

    const divisor = M[col][col];
    for (let j = col; j <= n; j += 1) M[col][j] /= divisor;

    for (let row = 0; row < n; row += 1) {
      if (row === col) continue;
      const factor = M[row][col];
      for (let j = col; j <= n; j += 1) M[row][j] -= factor * M[col][j];
    }
  }

  return M.map((row) => row[n]);
}

function combinations(size, choose, limit = 50000) {
  const result = [];
  const current = [];

  function visit(start) {
    if (result.length > limit) return;
    if (current.length === choose) {
      result.push([...current]);
      return;
    }

    for (let i = start; i <= size - (choose - current.length); i += 1) {
      current.push(i);
      visit(i + 1);
      current.pop();
    }
  }

  visit(0);
  return result;
}

function parseConstraint(line, expectedSize) {
  const match = line.match(/(.+?)(<=|>=|=)(.+)/);
  if (!match) {
    throw new Error(`Invalid constraint: "${line}". Use <=, >=, or =.`);
  }

  const coeffs = parseNumbers(match[1], 'Constraint coefficients');
  if (coeffs.length !== expectedSize) {
    throw new Error(`Constraint "${line}" has ${coeffs.length} coefficients, expected ${expectedSize}.`);
  }

  const rhs = Number(match[3].trim());
  if (!Number.isFinite(rhs)) throw new Error(`Invalid RHS in constraint: "${line}".`);

  return {
    coeffs,
    sense: match[2],
    rhs,
    label: line.trim(),
  };
}

function isFeasible(x, constraints) {
  const tol = 1e-7;
  if (x.some((value) => value < -tol)) return false;

  return constraints.every((constraint) => {
    const lhs = dot(constraint.coeffs, x);
    if (constraint.sense === '<=') return lhs <= constraint.rhs + tol;
    if (constraint.sense === '>=') return lhs >= constraint.rhs - tol;
    return Math.abs(lhs - constraint.rhs) <= tol;
  });
}

function solveLp() {
  const mode = $('#lpMode').value;
  const objective = parseNumbers($('#lpObjective').value, 'Objective coefficients');
  const names = $('#lpNames').value
    .split(',')
    .map((name) => name.trim())
    .filter(Boolean);

  const constraintLines = $('#lpConstraints').value
    .split('\n')
    .map((line) => line.trim())
    .filter(Boolean);

  if (!constraintLines.length) throw new Error('Add at least one constraint.');
  if (objective.length > 8) throw new Error('The browser LP solver supports up to 8 variables for safety.');

  const constraints = constraintLines.map((line) => parseConstraint(line, objective.length));
  const activeEquations = constraints.map((constraint, index) => ({
    coeffs: constraint.coeffs,
    rhs: constraint.rhs,
    label: `constraint ${index + 1}: ${constraint.label}`,
  }));

  for (let i = 0; i < objective.length; i += 1) {
    activeEquations.push({
      coeffs: objective.map((_, index) => (index === i ? 1 : 0)),
      rhs: 0,
      label: `${names[i] || `x${i}`} = 0`,
    });
  }

  const combos = combinations(activeEquations.length, objective.length);
  if (combos.length > 50000) {
    throw new Error('Too many possible vertices. Reduce the number of variables or constraints.');
  }

  const candidates = [];
  const seen = new Set();

  for (const combo of combos) {
    const A = combo.map((index) => activeEquations[index].coeffs);
    const b = combo.map((index) => activeEquations[index].rhs);
    const x = solveLinearSystem(A, b);
    if (!x || !isFeasible(x, constraints)) continue;

    const key = x.map((value) => value.toFixed(8)).join('|');
    if (seen.has(key)) continue;
    seen.add(key);

    const value = dot(objective, x);
    candidates.push({ x, value, combo });
  }

  if (!candidates.length) {
    throw new Error('No feasible vertex found. Check whether the problem is infeasible or incorrectly entered.');
  }

  candidates.sort((a, b) => (mode === 'max' ? b.value - a.value : a.value - b.value));
  const best = candidates[0];
  const active = constraints
    .map((constraint, index) => ({ constraint, index, lhs: dot(constraint.coeffs, best.x) }))
    .filter(({ constraint, lhs }) => Math.abs(lhs - constraint.rhs) <= 1e-6);

  const lines = [];
  lines.push('Status: OK');
  lines.push(`Method: vertex enumeration for small browser LPs`);
  lines.push(`Mode: ${mode === 'max' ? 'maximize' : 'minimize'}`);
  lines.push(`Objective: ${formatNumber(best.value)}`);
  lines.push('');
  lines.push('Variables:');
  best.x.forEach((value, index) => {
    lines.push(`  ${names[index] || `x${index}`} = ${formatNumber(value)}`);
  });
  lines.push('');
  lines.push(`Feasible vertices checked: ${candidates.length}`);
  lines.push('Active constraints at optimum:');
  if (active.length) {
    active.forEach(({ index, constraint, lhs }) => {
      lines.push(`  ${index + 1}. ${constraint.label}  [lhs=${formatNumber(lhs)}]`);
    });
  } else {
    lines.push('  none');
  }

  return lines.join('\n');
}

function solveKnapsack() {
  const values = parseNumbers($('#ksValues').value, 'Item values');
  const weights = parseNumbers($('#ksWeights').value, 'Item weights');
  const capacity = Number($('#ksCapacity').value);

  if (values.length !== weights.length) throw new Error('Values and weights must have the same length.');
  if (values.length > 24) throw new Error('The browser knapsack solver supports up to 24 items.');
  if (!Number.isFinite(capacity) || capacity < 0) throw new Error('Capacity must be a non-negative number.');

  const totalMasks = 1 << values.length;
  let bestValue = -Infinity;
  let bestWeight = 0;
  let bestMask = 0;

  for (let mask = 0; mask < totalMasks; mask += 1) {
    let totalValue = 0;
    let totalWeight = 0;

    for (let i = 0; i < values.length; i += 1) {
      if ((mask & (1 << i)) !== 0) {
        totalValue += values[i];
        totalWeight += weights[i];
      }
    }

    if (totalWeight <= capacity && totalValue > bestValue) {
      bestValue = totalValue;
      bestWeight = totalWeight;
      bestMask = mask;
    }
  }

  const lines = [];
  lines.push('Status: OK');
  lines.push('Method: exhaustive 0/1 search');
  lines.push(`Best value: ${formatNumber(bestValue)}`);
  lines.push(`Total weight: ${formatNumber(bestWeight)} / ${formatNumber(capacity)}`);
  lines.push('');
  lines.push('Selected items:');

  values.forEach((value, index) => {
    const chosen = (bestMask & (1 << index)) !== 0;
    lines.push(`  item ${index}: ${chosen ? 'take' : 'skip'}  value=${formatNumber(value)} weight=${formatNumber(weights[index])}`);
  });

  return lines.join('\n');
}

function rosenbrock([x, y]) {
  return (1 - x) ** 2 + 100 * (y - x * x) ** 2;
}

function rosenbrockGrad([x, y]) {
  return [
    -2 * (1 - x) - 400 * x * (y - x * x),
    200 * (y - x * x),
  ];
}

function matVec2(M, v) {
  return [M[0][0] * v[0] + M[0][1] * v[1], M[1][0] * v[0] + M[1][1] * v[1]];
}

function solveRosenbrock() {
  let x = [Number($('#rbX').value), Number($('#rbY').value)];
  const maxIterations = Number($('#rbIterations').value);
  const tolerance = Number($('#rbTolerance').value);

  if (x.some((value) => !Number.isFinite(value))) throw new Error('Start x and y must be numbers.');
  if (!Number.isInteger(maxIterations) || maxIterations <= 0) throw new Error('Max iterations must be a positive integer.');
  if (!Number.isFinite(tolerance) || tolerance <= 0) throw new Error('Tolerance must be a positive number.');

  let H = [[1, 0], [0, 1]];
  let gradient = rosenbrockGrad(x);
  let iterations = 0;

  for (; iterations < maxIterations; iterations += 1) {
    if (norm2(gradient) < tolerance) break;

    const Hg = matVec2(H, gradient);
    const direction = [-Hg[0], -Hg[1]];
    const directionalDerivative = dot(gradient, direction);
    let alpha = 1;
    const currentValue = rosenbrock(x);

    while (alpha > 1e-12) {
      const trial = [x[0] + alpha * direction[0], x[1] + alpha * direction[1]];
      if (rosenbrock(trial) <= currentValue + 1e-4 * alpha * directionalDerivative) break;
      alpha *= 0.5;
    }

    if (alpha <= 1e-12) break;

    const nextX = [x[0] + alpha * direction[0], x[1] + alpha * direction[1]];
    const nextGradient = rosenbrockGrad(nextX);
    const s = [nextX[0] - x[0], nextX[1] - x[1]];
    const y = [nextGradient[0] - gradient[0], nextGradient[1] - gradient[1]];
    const ys = dot(y, s);

    if (ys > 1e-12) {
      const rho = 1 / ys;
      const IminusRhoSY = [
        [1 - rho * s[0] * y[0], -rho * s[0] * y[1]],
        [-rho * s[1] * y[0], 1 - rho * s[1] * y[1]],
      ];
      const IminusRhoYS = [
        [1 - rho * y[0] * s[0], -rho * y[0] * s[1]],
        [-rho * y[1] * s[0], 1 - rho * y[1] * s[1]],
      ];
      const temp = multiply2(IminusRhoSY, H);
      H = add2(multiply2(temp, IminusRhoYS), [
        [rho * s[0] * s[0], rho * s[0] * s[1]],
        [rho * s[1] * s[0], rho * s[1] * s[1]],
      ]);
    }

    x = nextX;
    gradient = nextGradient;
  }

  const lines = [];
  lines.push('Status: OK');
  lines.push('Method: BFGS with Armijo backtracking');
  lines.push(`Objective: ${rosenbrock(x).toFixed(12)}`);
  lines.push(`x: [${x[0].toFixed(12)}, ${x[1].toFixed(12)}]`);
  lines.push(`Gradient norm: ${norm2(gradient).toExponential(6)}`);
  lines.push(`Iterations: ${iterations}`);
  return lines.join('\n');
}

function multiply2(A, B) {
  return [
    [A[0][0] * B[0][0] + A[0][1] * B[1][0], A[0][0] * B[0][1] + A[0][1] * B[1][1]],
    [A[1][0] * B[0][0] + A[1][1] * B[1][0], A[1][0] * B[0][1] + A[1][1] * B[1][1]],
  ];
}

function add2(A, B) {
  return [
    [A[0][0] + B[0][0], A[0][1] + B[0][1]],
    [A[1][0] + B[1][0], A[1][1] + B[1][1]],
  ];
}

function setOutput(title, text, ok = true) {
  outputTitle.textContent = title;
  output.textContent = text;
  output.style.color = ok ? '#dcfce7' : '#fecdd3';
}

function updateSolverForm() {
  const selected = solverSelect.value;
  $$('.solver-form').forEach((form) => form.classList.remove('active'));
  $(`#${selected}Form`).classList.add('active');
  setOutput('Ready', 'Edit the input and click solve.');
}

function loadExample(type) {
  if (type === 'diet') {
    solverSelect.value = 'lp';
    updateSolverForm();
    $('#lpMode').value = 'min';
    $('#lpNames').value = 'oats, milk, eggs';
    $('#lpObjective').value = '2, 1, 0.5';
    $('#lpConstraints').value = '10, 8, 6 >= 50\n300, 150, 70 >= 1500';
    setOutput('Diet example loaded', 'Click Solve LP to calculate the minimum cost diet.');
  }

  if (type === 'transport') {
    solverSelect.value = 'lp';
    updateSolverForm();
    $('#lpMode').value = 'min';
    $('#lpNames').value = 'x00, x01, x02, x10, x11, x12';
    $('#lpObjective').value = '2, 4, 5, 3, 1, 7';
    $('#lpConstraints').value = [
      '1, 1, 1, 0, 0, 0 <= 35',
      '0, 0, 0, 1, 1, 1 <= 50',
      '1, 0, 0, 1, 0, 0 >= 30',
      '0, 1, 0, 0, 1, 0 >= 30',
      '0, 0, 1, 0, 0, 1 >= 25',
    ].join('\n');
    setOutput('Transport example loaded', 'Click Solve LP to calculate the minimum transport cost.');
  }
}

solverSelect.addEventListener('change', updateSolverForm);

solverForm.addEventListener('submit', (event) => {
  event.preventDefault();

  try {
    let result;
    let title;

    if (solverSelect.value === 'lp') {
      result = solveLp();
      title = 'LP solved';
    } else if (solverSelect.value === 'knapsack') {
      result = solveKnapsack();
      title = 'Knapsack solved';
    } else {
      result = solveRosenbrock();
      title = 'BFGS complete';
    }

    setOutput(title, result);
  } catch (error) {
    setOutput('Error', error.message, false);
  }
});

$$('[data-example]').forEach((button) => {
  button.addEventListener('click', () => loadExample(button.dataset.example));
});

copyOutput.addEventListener('click', async () => {
  try {
    await navigator.clipboard.writeText(output.textContent);
    copyOutput.textContent = 'Copied';
    setTimeout(() => {
      copyOutput.textContent = 'Copy';
    }, 1200);
  } catch {
    setOutput('Copy failed', 'Your browser blocked clipboard access.', false);
  }
});

updateSolverForm();
