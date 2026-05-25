import fs from 'node:fs';
import vm from 'node:vm';

const elements = new Map();

function element(id, value = undefined) {
  if (!elements.has(id)) {
    elements.set(id, {
      value: value ?? '',
      textContent: '',
      style: {},
      dataset: {},
      classList: {
        add() {},
        remove() {},
      },
      addEventListener() {},
    });
  }
  const existing = elements.get(id);
  if (value !== undefined) {
    existing.value = value;
  }
  return existing;
}

[
  'solverSelect',
  'solverForm',
  'output',
  'outputTitle',
  'copyOutput',
  'lpMode',
  'lpNames',
  'lpObjective',
  'lpConstraints',
  'lpForm',
].forEach((id) => element(id));

elements.get('solverSelect').value = 'lp';

const context = {
  console,
  setTimeout,
  navigator: {
    clipboard: {
      async writeText() {},
    },
  },
  document: {
    querySelector(selector) {
      if (selector.startsWith('#')) {
        return element(selector.slice(1));
      }
      return element(selector);
    },
    querySelectorAll() {
      return [];
    },
  },
};

vm.createContext(context);
vm.runInContext(fs.readFileSync('public/app.js', 'utf8'), context);

function solveLp({ mode, objective, constraints, names = 'x, y' }) {
  elements.get('solverSelect').value = 'lp';
  elements.get('lpMode').value = mode;
  elements.get('lpNames').value = names;
  elements.get('lpObjective').value = objective;
  elements.get('lpConstraints').value = constraints;
  return vm.runInContext('solveLp()', context);
}

function assert(condition, message) {
  if (!condition) {
    throw new Error(message);
  }
}

const bounded = solveLp({
  mode: 'max',
  objective: '3, 2',
  constraints: '1, 1 <= 4\n1, 0 <= 2\n0, 1 <= 3',
});
assert(bounded.includes('Objective: 10.000000'), 'bounded LP objective changed');

try {
  solveLp({
    mode: 'max',
    objective: '1, 0',
    constraints: '0, 1 <= 1',
  });
  throw new Error('unbounded LP was reported as solved');
} catch (error) {
  assert(
    error.message.includes('unbounded'),
    `unexpected unbounded LP error: ${error.message}`,
  );
}

console.log('public app tests passed');
