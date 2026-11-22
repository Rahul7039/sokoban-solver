// Push-optimal Sokoban solver.
//
// Strategy: a level is a set of rooms off one hallway (the player's row). Cutting the dead squares of
// that row splits the floor into parts; each part holding boxes is a room whose boxes can never leave
// it, so the level's push optimum is the sum of its rooms' optima. Each room is solved on its own with a
// breadth-first search over pushes (never moves), the other rooms' boxes standing as walls, and the
// rooms' move strings are concatenated with the player walking on from where the last room left it.
//
// The search expands states in order of pushes made, so the first solved state it meets is push-optimal.
// It prunes two deadlocks a box can never leave: a dead square (one from which no push, ignoring the
// other boxes, brings a box to any goal) and a frozen 2 x 2 block of walls and boxes with a box off its
// goal. Neither prunes a solvable state, so the optimum stands. Every budget below is a count of states,
// never the clock, so two runs write the same bytes.
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const I = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const O = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** Directions in a fixed order: up, down, left, right, as [dx, dy, walk letter, push letter]. */
const DIRS: [number, number, string, string][] = [
  [0, -1, 'u', 'U'],
  [0, 1, 'd', 'D'],
  [-1, 0, 'l', 'L'],
  [1, 0, 'r', 'R'],
];

/** The most states one search keeps before it gives up: work, never the clock. */
const MAX_STATES = 2_500_000;

interface Level {
  w: number;
  h: number;
  wall: Uint8Array;
  goals: number[];
  boxes: number[];
  player: number;
}

/** Parse XSB level text; squares outside the rows and on the outermost frame count as walls. */
function parseLevel(text: string): Level {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const h = rows.length;
  const w = Math.max(...rows.map((r) => r.length));
  const wall = new Uint8Array(w * h).fill(1);
  const goals: number[] = [];
  const boxes: number[] = [];
  let player = -1;
  rows.forEach((row, y) => {
    for (let x = 0; x < row.length; x++) {
      const ch = row[x];
      const i = y * w + x;
      if (ch === '#') continue;
      wall[i] = 0;
      if (ch === '.' || ch === '*' || ch === '+') goals.push(i);
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  });
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (x === 0 || y === 0 || x === w - 1 || y === h - 1) wall[y * w + x] = 1;
    }
  }
  boxes.sort((a, b) => a - b);
  goals.sort((a, b) => a - b);
  return { w, h, wall, goals, boxes, player };
}

/**
 * The squares from which a box alone can be pushed onto some goal: pulled back from the goals over open
 * floor (a box moves from y along d when the landing square y + d and the player's square y - d are both
 * open). A box on any other square can never reach a goal.
 */
function liveSquares(level: Level): Uint8Array {
  const { w, wall, goals } = level;
  const live = new Uint8Array(wall.length);
  const queue = [...goals];
  for (const g of goals) live[g] = 1;
  for (let qi = 0; qi < queue.length; qi++) {
    const x = queue[qi];
    for (const [dx, dy] of DIRS) {
      const d = dy * w + dx;
      const y = x - d;
      if (!wall[y] && !wall[y - d] && !live[y]) {
        live[y] = 1;
        queue.push(y);
      }
    }
  }
  return live;
}

/** Mark the squares the player reaches from `from` with `occupied` in the way; answer the lowest one. */
function reach(level: Level, occupied: Uint8Array, from: number, seen: Int32Array, stamp: number, stack: Int32Array): number {
  const { w, wall } = level;
  let low = from;
  let n = 0;
  stack[n++] = from;
  seen[from] = stamp;
  while (n > 0) {
    const p = stack[--n];
    if (p < low) low = p;
    for (let k = 0; k < 4; k++) {
      const q = k === 0 ? p - w : k === 1 ? p + w : k === 2 ? p - 1 : p + 1;
      if (wall[q] === 0 && occupied[q] === 0 && seen[q] !== stamp) {
        seen[q] = stamp;
        stack[n++] = q;
      }
    }
  }
  return low;
}

/** Whether the box just pushed to `t` sits in a frozen 2 x 2 block of walls and boxes with a box off its goal. */
function frozenBlock(level: Level, occupied: Uint8Array, goal: Uint8Array, t: number): boolean {
  const { w, wall } = level;
  for (const a of [t, t - 1, t - w, t - w - 1]) {
    const b = a + 1;
    const c = a + w;
    const d = c + 1;
    if (!(wall[a] || occupied[a]) || !(wall[b] || occupied[b]) || !(wall[c] || occupied[c]) || !(wall[d] || occupied[d])) continue;
    if ((occupied[a] && !goal[a]) || (occupied[b] && !goal[b]) || (occupied[c] && !goal[c]) || (occupied[d] && !goal[d])) return true;
  }
  return false;
}

/** The shortest walk from `from` to `to` around walls and boxes, in lower LURD letters; null when out of reach. */
function walkTo(level: Level, occupied: Uint8Array, from: number, to: number): string | null {
  const { w, wall } = level;
  if (from === to) return '';
  const prev = new Int32Array(wall.length).fill(-1);
  const how = new Array<string>(wall.length);
  const queue = [from];
  prev[from] = from;
  for (let qi = 0; qi < queue.length; qi++) {
    const p = queue[qi];
    if (p === to) break;
    for (const [dx, dy, letter] of DIRS) {
      const q = p + dy * w + dx;
      if (wall[q] || occupied[q] || prev[q] !== -1) continue;
      prev[q] = p;
      how[q] = letter;
      queue.push(q);
    }
  }
  if (prev[to] === -1) return null;
  let s = '';
  for (let p = to; p !== from; p = prev[p]) s = how[p] + s;
  return s;
}

/** The square the player stands on after playing `moves` from `from`. */
function endOf(w: number, from: number, moves: string): number {
  let p = from;
  for (const ch of moves.toLowerCase()) p += ch === 'u' ? -w : ch === 'd' ? w : ch === 'l' ? -1 : 1;
  return p;
}

interface Solution {
  pushes: number;
  moves: string;
}

/**
 * A push-optimal solution of one (sub-)level by breadth-first search over pushes, or null when the
 * search gives up or the level cannot be solved. A state is the boxes' squares plus the player's
 * reachable region, named by its lowest square; a step is one push, with the walk to its square worked
 * out when the path is rebuilt.
 */
function solvePushOptimal(level: Level): Solution | null {
  const { w, wall, goals } = level;
  const size = wall.length;
  const goal = new Uint8Array(size);
  for (const g of goals) goal[g] = 1;
  const live = liveSquares(level);
  const solved = (bs: number[]): boolean => bs.every((b) => goal[b]);
  if (level.boxes.some((b) => !live[b])) return null;

  const occupied = new Uint8Array(size);
  const seen = new Int32Array(size);
  const mark = new Int32Array(size);
  const stack = new Int32Array(size);
  let stamp = 0;
  // A state's key: the region's lowest square and the boxes' squares. Digits in base `size` while that
  // stays exact (a room of up to five boxes on any board this task draws), else a string of char codes.
  const numeric = (level.boxes.length + 1) * Math.log2(Math.max(size, 2)) < 53;
  const keyOf = (rep: number, bs: number[]): number | string => {
    if (numeric) {
      let k = rep;
      for (const b of bs) k = k * size + b;
      return k;
    }
    return String.fromCharCode(rep, ...bs);
  };

  // The states in the order found: their boxes, the player's square, and the parent state.
  const statesBoxes: number[][] = [level.boxes];
  const statesPlayer: number[] = [level.player];
  const parent: number[] = [-1];
  const index = new Set<number | string>();
  for (const b of level.boxes) occupied[b] = 1;
  index.add(keyOf(reach(level, occupied, level.player, seen, ++stamp, stack), level.boxes));
  for (const b of level.boxes) occupied[b] = 0;
  let found = solved(level.boxes) ? 0 : -1;
  for (let head = 0; found < 0 && head < statesBoxes.length; head++) {
    const boxes = statesBoxes[head];
    for (const b of boxes) occupied[b] = 1;
    reach(level, occupied, statesPlayer[head], seen, ++stamp, stack);
    const here = stamp;
    for (let bi = 0; bi < boxes.length && found < 0; bi++) {
      const b = boxes[bi];
      for (const [dx, dy] of DIRS) {
        const d = dy * w + dx;
        const t = b + d;
        if (seen[b - d] !== here || wall[t] || occupied[t] || !live[t]) continue;
        occupied[b] = 0;
        occupied[t] = 1;
        const frozen = frozenBlock(level, occupied, goal, t);
        let key: number | string = -1;
        let next: number[] | null = null;
        if (!frozen) {
          next = boxes.slice();
          next[bi] = t;
          next.sort((x, y) => x - y);
          key = keyOf(reach(level, occupied, b, mark, ++stamp, stack), next);
        }
        occupied[t] = 0;
        occupied[b] = 1;
        if (frozen || index.has(key)) continue;
        if (statesBoxes.length >= MAX_STATES) return null;
        index.add(key);
        statesBoxes.push(next as number[]);
        statesPlayer.push(b);
        parent.push(head);
        if (solved(next as number[])) found = statesBoxes.length - 1;
      }
    }
    for (const b of boxes) occupied[b] = 0;
  }
  if (found < 0) return null;

  // The path of states, then the moves: walk to each push's square, then push.
  const path: number[] = [];
  for (let s = found; s > 0; s = parent[s]) path.push(s);
  path.reverse();
  let moves = '';
  let player = level.player;
  let boxes = level.boxes;
  for (const s of path) {
    const nextBoxes = statesBoxes[s];
    const from = boxes.find((b) => !nextBoxes.includes(b)) as number;
    const to = nextBoxes.find((b) => !boxes.includes(b)) as number;
    const d = to - from;
    const dir = DIRS.find(([dx, dy]) => dy * w + dx === d) as [number, number, string, string];
    for (const b of boxes) occupied[b] = 1;
    const walk = walkTo(level, occupied, player, from - d);
    for (const b of boxes) occupied[b] = 0;
    if (walk === null) return null;
    moves += walk + dir[3];
    player = from;
    boxes = nextBoxes;
  }
  return { pushes: path.length, moves };
}

interface Room {
  squares: number[];
  boxes: number[];
  goals: number[];
}

/**
 * The level's rooms: cut the dead squares of the player's row (the hallway: a box there reaches no goal)
 * out of the floor, and each part left that holds a box is a room. Null when the floor does not split
 * into rooms that way: some part holding more or fewer boxes than goals, a goal outside every part, or
 * a box starting in the hallway.
 */
function roomsOf(level: Level): Room[] | null {
  const { w, wall, goals, boxes, player } = level;
  const live = liveSquares(level);
  const row = Math.floor(player / w);
  const cut = new Uint8Array(wall.length);
  for (let i = row * w; i < (row + 1) * w; i++) if (!wall[i] && !live[i]) cut[i] = 1;
  if (boxes.some((b) => cut[b])) return null;
  const part = new Int32Array(wall.length).fill(-1);
  const rooms: Room[] = [];
  for (let s = 0; s < wall.length; s++) {
    if (wall[s] || cut[s] || part[s] >= 0) continue;
    const squares = [s];
    part[s] = rooms.length;
    for (let i = 0; i < squares.length; i++) {
      for (const [dx, dy] of DIRS) {
        const q = squares[i] + dy * w + dx;
        if (!wall[q] && !cut[q] && part[q] < 0) {
          part[q] = rooms.length;
          squares.push(q);
        }
      }
    }
    rooms.push({ squares, boxes: [], goals: [] });
  }
  for (const b of boxes) rooms[part[b]].boxes.push(b);
  for (const g of goals) if (part[g] >= 0) rooms[part[g]].goals.push(g);
  if (goals.some((g) => part[g] < 0) || rooms.some((r) => r.boxes.length !== r.goals.length)) return null;
  return rooms.filter((r) => r.boxes.length > 0);
}

/**
 * Solve the level with the rooms taken in the given order: each room on a copy of the board where the
 * other rooms' boxes stand as walls (on their goals once their room is done), the player walking on
 * from where the last room's moves left it. Null when a room's search gives up.
 */
function solveInOrder(level: Level, rooms: Room[], order: number[]): Solution | null {
  let player = level.player;
  let pushes = 0;
  let moves = '';
  const done = new Set<number>();
  for (const ri of order) {
    const room = rooms[ri];
    const wall = level.wall.slice();
    rooms.forEach((other, oi) => {
      if (oi === ri) return;
      for (const b of done.has(oi) ? other.goals : other.boxes) wall[b] = 1;
    });
    const sub: Level = {
      w: level.w,
      h: level.h,
      wall,
      goals: room.goals.slice().sort((a, b) => a - b),
      boxes: room.boxes.slice().sort((a, b) => a - b),
      player,
    };
    const solvedRoom = solvePushOptimal(sub);
    if (solvedRoom === null) return null;
    pushes += solvedRoom.pushes;
    moves += solvedRoom.moves;
    player = endOf(level.w, player, solvedRoom.moves);
    done.add(ri);
  }
  return { pushes, moves };
}

/** Every order of the rooms, the natural one first, each order itself in a fixed sequence. */
function roomOrders(n: number): number[][] {
  const base = Array.from({ length: n }, (_, i) => i);
  if (n <= 1) return [base];
  const out: number[][] = [];
  const permute = (arr: number[], l: number): void => {
    if (l === arr.length) {
      out.push(arr.slice());
      return;
    }
    for (let i = l; i < arr.length; i++) {
      [arr[l], arr[i]] = [arr[i], arr[l]];
      permute(arr, l + 1);
      [arr[l], arr[i]] = [arr[i], arr[l]];
    }
  };
  permute(base.slice(), 0);
  return out;
}

/** The level's solution: room by room when it splits into rooms, else one search over the whole level. */
function solveLevel(level: Level): Solution | null {
  const rooms = roomsOf(level);
  if (rooms !== null) {
    for (const order of roomOrders(rooms.length)) {
      const solved = solveInOrder(level, rooms, order);
      if (solved !== null) return solved;
    }
    return null;
  }
  return solvePushOptimal(level);
}

function solve(text: string): string {
  const level = parseLevel(text);
  if (level.player < 0) return '';
  const solved = solveLevel(level);
  return solved === null ? '' : solved.moves;
}

for (const id of readdirSync(I).sort()) {
  const moves = solve(readFileSync(join(I, id, 'level.txt'), 'utf8'));
  mkdirSync(join(O, id), { recursive: true });
  writeFileSync(join(O, id, 'solution.txt'), `${moves}\n`);
}
