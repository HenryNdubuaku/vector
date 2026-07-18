# Vector by example

Generated from the test suite by `cargo test` — every program and its output are verified on each run. Do not edit by hand.

## algos

```python
x = [3.0, 1.0, 2.0]
print(sort(x))
print(argsort(x))
print(argmax(x))
print(argmin(x))
print(one_hot(1.0, 3))
print(one_hot([2.0, 0.0], 3))
print(take([10.0, 20.0, 30.0], [2.0, 0.0]))
print(cumsum([1.0, 2.0, 3.0]))
print(softmax([0.0, 0.0]))
print(relu([-1.0, 2.0]))
print(sigmoid(0.0))
print(std([1.0, 3.0]))
print(norm([3.0, 4.0]))
print(logsumexp([0.0, 0.0]))
print(where([1.0, 2.0] == [1.0, 3.0], 1.0, 0.0))
print(where([1.0, 2.0] != [1.0, 3.0], 1.0, 0.0))

m = [[3.0, 1.0], [0.0, 5.0]]
function am(r):
  argmax(r)

print(vmap(am, m))

function srt(r):
  sort(r)

print(vmap(srt, m))

function f(v):
  sum(cumsum(v))

print(grad(f, [1.0, 2.0, 3.0]))

function g(v):
  sum(take(v, [2.0, 2.0]))

print(grad(g, [1.0, 2.0, 3.0]))
```

Output:

```
[1, 2, 3] : f32
[1, 2, 0] : f32
0 : f32
1 : f32
[0, 1, 0] : f32
[[0, 0, 1], [1, 0, 0]] : f32
[30, 10] : f32
[1, 3, 6] : f32
[0.5, 0.5] : f32
[0, 2] : f32
0.5 : f32
1 : f32
5 : f32
0.6931472 : f32
[1, 0] : f32
[0, 1] : f32
[0, 1] : f32
[[1, 3], [0, 5]] : f32
[3, 2, 1] : f32
[0, 0, 2] : f32
```

## arithmetic

```python
# comments run to end of line
print(-3.0 + 2.0 * 5.0)  # including after expressions
print(-(1.0 + 2.0))
print(2.0 / -4.0)
print(1.0 - -1.0)
print(--5.0)
```

Output:

```
7 : f32
-3 : f32
-0.5 : f32
2 : f32
5 : f32
```

## attention

```python
function mm(env, w):
  matmul(env.x, w)

function qk(q, k):
  matmul(q, transpose(k))

function sm(row):
  softmax(row)

function rows_sm(m):
  vmap(sm, m)

function av(a, v):
  matmul(a, v)

function attend(p, x):
  q = vmap(mm, {x: x}, p.wq)
  k = vmap(mm, {x: x}, p.wk)
  v = vmap(mm, {x: x}, p.wv)
  scores = vmap(qk, q, k) * p.scale + p.mask
  att = vmap(rows_sm, scores)
  heads = vmap(av, att, v)
  sum(vmap(av, heads, p.wo), 0)

function xent(logits, target):
  logsumexp(logits) - sum(one_hot(target, 16) * logits)

function seq_loss(p, ids, targets):
  x = take(p.wte, ids) + p.wpe
  logits = matmul(attend(p, x), p.head)
  mean(vmap(xent, logits, targets))

function batch_loss(p, idsb, targetsb):
  mean(vmap(seq_loss, p, idsb, targetsb))

rows = matmul(reshape(arange(4.0), 4, 1), reshape(zeros(4) + 1.0, 1, 4))
p = {
  wte: reshape(sin(arange(128.0)), 16, 8) * 0.5,
  wpe: reshape(cos(arange(32.0)), 4, 8) * 0.1,
  wq: reshape(sin(arange(64.0)), 2, 8, 4) * 0.3,
  wk: reshape(cos(arange(64.0)), 2, 8, 4) * 0.3,
  wv: reshape(sin(arange(64.0) + 1.0), 2, 8, 4) * 0.3,
  wo: reshape(cos(arange(64.0) + 1.0), 2, 4, 8) * 0.3,
  head: reshape(sin(arange(128.0) + 2.0), 8, 16) * 0.3,
  scale: 0.5,
  mask: where(rows >= transpose(rows), 0.0, 0.0 - 1000000000.0)
}
idsb = stack([1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0])
targetsb = stack([2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0])

before = batch_loss(p, idsb, targetsb)
print(before)

g = grad(batch_loss, p, idsb, targetsb)
print(where(sum(g.wq * g.wq) > 0.0, 1.0, 0.0))
print(where(sum(g.wte * g.wte) > 0.0, 1.0, 0.0))

p2 = p - 0.01 * g / (sqrt(g * g) + 0.00000001)
after = batch_loss(p2, idsb, targetsb)
print(where(after < before, 1.0, 0.0))
```

Output:

```
2.7732534 : f32
1 : f32
1 : f32
1 : f32
```

## bpe

```python
# byte pair encoding from scratch: count pairs, merge the best, compact with a stable argsort

vmax = 260.0
junk = vmax * vmax - 1.0
ids = load("tests/cases/data/bpe.txt")
dead = zeros(11)

for round in 0..2:
  ids = take(ids, argsort(dead))
  dead = sort(dead)
  left = ids[:10]
  right = ids[1:]
  live = where(dead[:10] + dead[1:] == 0.0, 1.0, 0.0)
  codes = live * (left * vmax + right) + (1.0 - live) * junk
  counts = bincount(codes, 67600) * (1.0 - one_hot(junk, 67600))
  best = argmax(counts)
  a = floor(best / vmax)
  b = mod(best, vmax)
  print(a)
  print(b)
  hit = where(left == a, 1.0, 0.0) * where(right == b, 1.0, 0.0) * live
  hit = hit * (1.0 - concat([0.0], hit[:9]))
  ids = where(concat(hit, [0.0]) == 1.0, vmax - 4.0 + round, ids)
  dead = maximum(dead, concat([0.0], hit))

ids = take(ids, argsort(dead))
dead = sort(dead)
print(where(dead == 1.0, 0.0 - 1.0, ids))
print(sum(1.0 - dead))
```

Output:

```
round 0: 97 : f32
round 1: 256 : f32
round 0: 98 : f32
round 1: 256 : f32
[257, 256, 99, 257, -1, -1, -1, -1, -1, -1, -1] : f32
4 : f32
```

## broadcast

```python
m = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
b = [10.0, 20.0, 30.0]
print(m + b)
print(m * 2.0)
```

Output:

```
[[11, 22, 33], [14, 25, 36]] : f32
[[2, 4, 6], [8, 10, 12]] : f32
```

## conv

```python
img = reshape(arange(9.0), 3, 3, 1)
k = reshape(zeros(4) + 1.0, 2, 2, 1, 1)
print(reshape(conv(img, k, 1, 0), 2, 2))

function loss(kernel, image):
  sum(conv(image, kernel, 1, 0))

g = grad(loss, k, img)
print(reshape(g, 2, 2))

function loss2(image, kernel):
  sum(conv(image, kernel, 1, 0) * conv(image, kernel, 1, 0))

g2 = grad(loss2, img, k)
print(sum(g2))

c = Conv(3, 1, 4, 2)
print(len(c(reshape(arange(64.0), 8, 8, 1))))
```

Output:

```
[[8, 12], [20, 24]] : f32
[[8, 12], [20, 24]] : f32
512 : f32
4 : f32
```

## dft

```python
wave = cos(arange(64.0) * (2.0 * pi * 4.0 / 64.0))
angles = matmul(reshape(arange(16.0), 16, 1), reshape(arange(64.0), 1, 64)) * (2.0 * pi / 64.0)
re = matmul(reshape(wave, 1, 64), transpose(cos(angles)))
im = matmul(reshape(wave, 1, 64), transpose(sin(angles)))
power = re * re + im * im
print(argmax(power[0]))
print(where(power[0][4] > 100.0 * power[0][7], 1.0, 0.0))
```

Output:

```
4 : f32
1 : f32
```

## dtypes

```python
xs = f32([3.0, 4.0])
print(xs + 0.5)
print(f64(xs))
```

Output:

```
[3.5, 4.5] : f32
[3, 4] : f64
```

## for_where

```python
w = 1.0
for i in 0..4:
  w = w * 2.0
print(w)

acc = 0.0
for i in 0..5:
  acc = acc + i
print(acc)

function sq_loss(v):
  d = v - 3.0
  sum(d * d)

v = 0.0
for i in 0..20:
  v = v - 0.5 * grad(sq_loss, v)
print(v)

x = [-1.0, 0.5, 2.0]
print(where(x > 0.0, x, 0.0))
print(where(x < 1.0, 1.0, -1.0))

function hinge_like(h):
  sum(where(h > 0.0, h, h * 0.0))

print(grad(hinge_like, [2.0, -3.0]))
```

Output:

```
16 : f32
10 : f32
3 : f32
[0, 0.5, 2] : f32
[1, 1, -1] : f32
[1, 0] : f32
```

## functions

```python
function norm_sq(xs):
  squares = xs * xs
  sum(squares)

xs = f32([3.0, 4.0])
print(norm_sq(xs))
```

Output:

```
25 : f32
```

## gather

```python
e = reshape(arange(32), 8, 4)
print(take(e, [2.0, 0.0, 2.0]))
print(take(e, 3.0))
print(take([10.0, 20.0, 30.0], 1.0))

function f(w):
  sum(take(w, [1.0, 1.0, 3.0]))

print(grad(f, reshape(arange(32), 8, 4)))

function s(v):
  sum(sort(v) * [1.0, 2.0, 3.0])

print(grad(s, [3.0, 1.0, 2.0]))

function pick(r):
  take(r, [1.0, 0.0])

print(vmap(pick, [[10.0, 20.0], [30.0, 40.0]]))
```

Output:

```
[[8, 9, 10, 11], [0, 1, 2, 3], [8, 9, 10, 11]] : f32
[12, 13, 14, 15] : f32
20 : f32
[[0, 0, 0, 0], [2, 2, 2, 2], [0, 0, 0, 0], [1, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]] : f32
[3, 1, 2] : f32
[[20, 10], [40, 30]] : f32
```

## generate

```python
print(arange(5))
print(arange(2, 5))
print(arange(1.0, 2.0, 0.25))
print(arange(3, 0, -1))
print(reshape(arange(6), 2, 3))
print(sin(0.0))
print(cos(0.0))

function s(x):
  sum(sin(x))

print(grad(s, [0.0, 0.0]))

function r(x):
  sum(reshape(x, 4) * [1.0, 2.0, 3.0, 4.0])

print(grad(r, [[1.0, 1.0], [1.0, 1.0]]))
print(linspace(0.0, 1.0, 5))
half = 0.5
print(linspace(-half, half, 3))
n = 4.0
print(arange(n))
print(linspace(-pi, pi, 3))
print(sin(pi / 2.0))
```

Output:

```
[0, 1, 2, 3, 4] : f32
[2, 3, 4] : f32
[1, 1.25, 1.5, 1.75] : f32
[3, 2, 1] : f32
[[0, 1, 2], [3, 4, 5]] : f32
0 : f32
1 : f32
[1, 1] : f32
[[1, 2], [3, 4]] : f32
[0, 0.25, 0.5, 0.75, 1] : f32
[-0.5, 0, 0.5] : f32
[0, 1, 2, 3] : f32
[-3.1415927, 0, 3.1415927] : f32
1 : f32
```

## grad

```python
function norm_sq(xs):
  sum(xs * xs)

print(grad(norm_sq, [3.0, 4.0]))

function hinge(x):
  sum(maximum(x, 0.0))

print(grad(hinge, [1.0, -2.0]))

function mean_sq(x):
  mean(x * x)

print(grad(mean_sq, [3.0, 6.0]))

function lin(w):
  sum(matmul([[1.0, 2.0], [3.0, 4.0]], w))

print(grad(lin, [[1.0, 1.0], [1.0, 1.0]]))

function chain(x):
  sum(sqrt(exp(x)))

print(grad(chain, [0.0]))

function recip(x):
  sum(1.0 / x)

print(grad(recip, [2.0]))

function biased(b):
  sum([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]] + b)

print(grad(biased, [0.0, 0.0, 0.0]))

function stacked(x):
  sum([x, x] * [2.0, 3.0])

print(grad(stacked, 5.0))

function first(w, b):
  sum(w * b)

print(grad(first, [1.0, 2.0], [3.0, 4.0]))

function times_five(x):
  sum(x * 5.0)

print(grad(times_five, [5.0]))
```

Output:

```
[6, 8] : f32
[1, 0] : f32
[3, 6] : f32
[[4, 4], [6, 6]] : f32
[0.5] : f32
[-0.25] : f32
[2, 2, 2] : f32
5 : f32
[3, 4] : f32
[5] : f32
```

## grad2

```python
function sq(x):
  sum(x * x)

function dsq(x):
  grad(sq, x)

print(grad(dsq, 3.0))

function stacked(x):
  sum([x, x] * [2.0, 3.0])

function dstacked(x):
  grad(stacked, x)

print(grad(dstacked, 5.0))

function cube(x):
  sum(x * x * x)

function dcube(x):
  grad(cube, x)

function ddcube(x):
  grad(dcube, x)

print(grad(dcube, 2.0))
print(grad(ddcube, 2.0))

function growth(x):
  sum(exp(x))

function dgrowth(x):
  grad(growth, x)

print(grad(dgrowth, 0.0))
```

Output:

```
2 : f32
0 : f32
12 : f32
6 : f32
1 : f32
```

## gz

```python
x = load("tests/cases/data/tiny.gz")
print(len(x))
print(x)
print(sum(x))
```

Output:

```
2 : f32
[[[0, 1], [2, 3]], [[4, 5], [6, 7]]] : f32
28 : f32
```

## index

```python
x = arange(10.0)
print(x[3.0])
print(x[2:5])
print(x[:3])
print(x[-3:])
print(x[[1.0, 4.0, 1.0]])

m = reshape(arange(6.0), 3, 2)
print(m[1])
print(m[1:3])

print(abs([0.0 - 3.0, 2.0]))
print(pow([2.0, 3.0], 2.0))
print(concat([1.0, 2.0], [3.0]))
print(stack([1.0, 2.0], [3.0, 4.0]))

function f(x):
  sum(x[1:3] * 2.0 + pow(x[0], 2.0))

print(grad(f, [1.0, 2.0, 3.0]))

print(len(x))
print(slice(x, [0.0, 3.0, 7.0], 3))
win = random_windows(x, 2, 4)
print(where(sum(abs(win[0][1:] - win[0][:3] - 1.0)) == 0.0, 1.0, 0.0))

s = sum([2.0])
print(x[s : s + 3])
print(x[s * 2.0 : s * 2.0 + 3])
```

Output:

```
3 : f32
[2, 3, 4] : f32
[0, 1, 2] : f32
[7, 8, 9] : f32
[1, 4, 1] : f32
[2, 3] : f32
[[2, 3], [4, 5]] : f32
[3, 2] : f32
[4, 9] : f32
[1, 2, 3] : f32
[[1, 2], [3, 4]] : f32
[4, 2, 2] : f32
10 : f32
[[0, 1, 2], [3, 4, 5], [7, 8, 9]] : f32
1 : f32
[2, 3, 4] : f32
[4, 5, 6] : f32
```

## jacobian

```python
function poly(x):
  x * x * 2.0

print(jacobian(poly, [1.0, 2.0]))

function affine(x):
  matmul([[1.0, 2.0], [3.0, 4.0]], x)

print(jacobian(affine, [5.0, 7.0]))

function jpoly(x):
  jacobian(poly, x)

print(vmap(jpoly, [[1.0, 2.0], [3.0, 1.0]]))
```

Output:

```
[[4, 0], [0, 8]] : f32
[[1, 2], [3, 4]] : f32
[[[4, 0], [0, 8]], [[12, 0], [0, 4]]] : f32
```

## linear

```python
layer = Linear(2, 3)
print(layer.b)
print(layer([0.0, 0.0]))
g = glorot_uniform(8, 8)
print(sum(where(g * g <= 0.375, 1.0, 0.0)))
h = he_normal(4, 4)
print(sum(h * 0.0))
print(sum(lecun_uniform(6, 2) * 0.0))
```

Output:

```
[0, 0, 0] : f32
[0, 0, 0] : f32
64 : f32
0 : f32
0 : f32
```

## load

```python
m = load("tests/cases/data/m.npy")
v = load("tests/cases/data/v.npy")
print(m + 0.5)
print(sum(v * v))
print(matmul(m, f32([1.0, 1.0])))

function q(x):
  sum(x * x)

print(grad(q, v))
```

Output:

```
[[2, 3], [4, 5]] : f32
25 : f64
[4, 8] : f32
[6, 8] : f64
```

## loop_print

```python
w = 1.0
for i in 0..3:
  w = w * 2.0
  print(w)
print(w)

acc = 0.0
for x in 0.0..1.0 by 0.5:
  acc = acc + x
  print({a: acc, b: x})
print(acc)

function f(v):
  y = v
  for k in 0..2:
    y = y * y
    print(sum(y))
  sum(y)

print(grad(f, [2.0]))
```

Output:

```
i 0: 2 : f32
i 1: 4 : f32
i 2: 8 : f32
8 : f32
x 0: a: 0 : f32
x 0.5: a: 0.5 : f32
x 0: b: 0 : f32
x 0.5: b: 0.5 : f32
0.5 : f32
k 0: 4 : f32
k 1: 16 : f32
[32] : f32
```

## loops

```python
function looped(x):
  y = x
  for i in 0..3:
    y = y * y
  sum(y)

print(grad(looped, [2.0]))

function inner_double(x):
  y = x
  for j in 0..2:
    y = y * 2.0
  y

acc = 1.0
for i in 0..3:
  acc = inner_double(acc)
print(acc)

state = {v: [1.0, 2.0], scale: 2.0}
for i in 0..3:
  state = state * state.scale / 2.0 + {v: [1.0, 1.0], scale: 1.0} * 0.0 + state * 0.0 + state
print(state.v)
acc = 0.0
for i in 0..10 by 2:
  acc = acc + i
print(acc)

for i in 5..0 by -1:
  acc = acc + i
print(acc)

for x in 0.0..1.0 by 0.25:
  acc = acc + x
print(acc)

function f(v):
  y = v
  for i in 0..4 by 2:
    y = y * 2.0
  sum(y)

print(grad(f, [3.0]))
```

Output:

```
[1024] : f32
64 : f32
[42, 84] : f32
20 : f32
35 : f32
36.5 : f32
[4] : f32
```

## math

```python
print(exp(0.0))
print(sqrt([4.0, 9.0]))
print(tanh(0.0))
print(log(1.0))
print(maximum(-1.0, [0.5, -2.0]))
print(minimum(2.0, 1.0))
```

Output:

```
1 : f32
[2, 3] : f32
0 : f32
0 : f32
[0.5, -1] : f32
1 : f32
```

## matmul

```python
a = [[1.0, 2.0], [3.0, 4.0]]
b = [[5.0, 6.0], [7.0, 8.0]]
print(matmul(a, b))
v = [1.0, 1.0]
print(matmul(a, v))
print(matmul(v, a))
print(matmul(v, v))
```

Output:

```
[[19, 22], [43, 50]] : f32
[3, 7] : f32
[4, 6] : f32
2 : f32
```

## modules

```python
module Scale(k):
  s = 0.0 + k
  forward(self, x):
    self.s * x

sc = Scale(3)
print(sc([1.0, 2.0]))
print(sc.s)

function l(m):
  sum(m([2.0]))

print(grad(l, sc))

sc2 = sc - 0.5 * grad(l, sc)
print(sc2.s)
print(sc2([1.0]))

module Z(n):
  v = zeros(n)
  forward(self, x):
    self.v + x

z = Z(3)
print(z(1.0))

module Pair():
  a = Scale(2)
  b = Scale(4)
  forward(self, x):
    self.b(self.a(x))

p = Pair()
print(p([1.0]))

function pl(m):
  sum(m([1.0]))

print(grad(pl, p))
print(sum(randn(2, 2) * 0.0))

module Quad(k):
  c = 0.0 + k
  forward(self, x):
    self.c * x
  energy(self):
    e = self.c - 3.0
    sum(e * e)
  scaled_energy(self, w):
    w * self.energy()

q = Quad(1)
print(q.energy())
print(grad(q.energy))
q2 = q - 0.25 * grad(q.energy)
print(q2.c)
print(q2.energy())
print(q2([2.0]))
print(q.scaled_energy(2.0))
print(grad(q.scaled_energy, 2.0))
```

Output:

```
[3, 6] : f32
3 : f32
s: 2 : f32
2 : f32
[2] : f32
[1, 1, 1] : f32
[8] : f32
a.s: 4 : f32
b.s: 2 : f32
0 : f32
4 : f32
c: -4 : f32
2 : f32
1 : f32
[4] : f32
8 : f32
c: -8 : f32
```

## nn

```python
print(mse([1.0, 2.0], [0.0, 4.0]))
print(cross_entropy([1.0, 2.0, 3.0], 2.0))
print(clip([-5.0, 0.5, 5.0], -1.0, 1.0))
print(clip_by_norm([3.0, 4.0], 1.0))
print(cosine_decay(1.0, 5.0, 10.0))
print(warmup(1.0, 5.0, 10.0))
print(layer_norm([1.0, 2.0, 3.0], 2.0, 1.0))
emb = Embedding(8, 4)
rows = emb([3.0, 3.0, 5.0])
print(sum(abs(rows[0] - rows[1])))
print(where(sum(abs(rows[0] - rows[2])) > 0.0, 1.0, 0.0))

u = normal(4096)
print(where(abs(mean(u)) < 0.1, 1.0, 0.0))
print(where(abs(std(u) - 1.0) < 0.1, 1.0, 0.0))

function loss(p, x):
  mse(p.w * x, x * 3.0)

st = adam_init({w: 0.0})
x = [1.0, 2.0]
n = 0.0
while mse(st.p.w * x, x * 3.0) > 0.01:
  st = adam(st, grad(loss, st.p, x), 0.1)
  n = n + 1.0
print(where(n < 500.0, 1.0, 0.0))
print(where(abs(st.p.w - 3.0) < 0.2, 1.0, 0.0))

s2 = sgd_init({w: 0.0})
for i in 0..50:
  s2 = sgd(s2, grad(loss, s2.p, x), 0.05, 0.9)
print(where(abs(s2.p.w - 3.0) < 0.2, 1.0, 0.0))

sm = softmax(reshape(arange(6.0), 2, 3))
print(sum(sm, 1))
bm = matmul(reshape(arange(8.0), 2, 2, 2), reshape(arange(8.0), 2, 2, 2))
print(bm)
print(sum(matmul(reshape(arange(8.0), 2, 2, 2), reshape(arange(4.0), 2, 2))))
```

Output:

```
2.5 : f32
0.4076059 : f32
[-1, 0.5, 1] : f32
[0.6, 0.8] : f32
0.49999997 : f32
0.5 : f32
[-1.4494712, 1, 3.4494712] : f32
0 : f32
1 : f32
1 : f32
1 : f32
1 : f32
1 : f32
1 : f32
[1, 1] : f32
[[[2, 3], [6, 11]], [[46, 55], [66, 79]]] : f32
92 : f32
```

## optimizer

```python
function f(x):
  sum(x * x)

state = {x: [1.0], m: [0.0]}
for i in 0..3:
  g = grad(f, state.x)
  m = 0.5 * state.m + g
  state = {x: state.x - 0.5 * m, m: m}
print(state.x)
print(state.m)
```

Output:

```
[-0.25] : f32
[-0.5] : f32
```

## random

```python
x = zeros(4096) + 1.0
m = mean(dropout(x, 0.5))
print(where(m > 0.85, 1.0, 0.0) * where(m < 1.15, 1.0, 0.0))

u = uniform(4096)
print(where(mean(u) > 0.4, 1.0, 0.0) * where(mean(u) < 0.6, 1.0, 0.0))
print(where(max(u) <= 1.0, 1.0, 0.0) * where(min(u) >= 0.0, 1.0, 0.0))

function pick(r):
  sample(r * 1.0)

logits = zeros(1000, 3) + [0.0, 10.0, 0.0]
hits = mean(where(vmap(pick, logits) == 1.0, 1.0, 0.0))
print(where(hits > 0.95, 1.0, 0.0))

prev = zeros(16)
repeats = 0.0
for i in 0..8:
  u2 = uniform(16)
  repeats = repeats + sum(where(u2 == prev, 1.0, 0.0))
  prev = u2 * 1.0
print(repeats)

p = permutation(64)
print(sum(where(sort(p) == arange(64), 1.0, 0.0)))
print(where(sum(where(p == arange(64), 1.0, 0.0)) < 64.0, 1.0, 0.0))

function dloss(w):
  mean(dropout(w, 0.25) * w)

g = grad(dloss, zeros(64) + 2.0)
print(where(sum(g * g) > 0.0, 1.0, 0.0))
```

Output:

```
1 : f32
1 : f32
1 : f32
1 : f32
0 : f32
64 : f32
1 : f32
1 : f32
```

## records

```python
p = {a: [1.0, 2.0], b: 3.0}
print(p)
print(p.a + p.b)
q = p * 2.0
print(q.a)
r = p + q
print(r.b)

function nsq(p):
  sum(p.a * p.a) + p.b * p.b

print(grad(nsq, p))

nested = {inner: {v: [1.0, 1.0]}, w: 2.0}
print(nested.inner.v)

function nloss(n):
  sum(n.inner.v * n.inner.v) * n.w

print(grad(nloss, nested))

s = {x: 1.0}
for i in 0..3:
  s = s + s
print(s.x)
```

Output:

```
a: [1, 2] : f32
b: 3 : f32
[4, 5] : f32
[2, 4] : f32
9 : f32
a: [2, 4] : f32
b: 6 : f32
[1, 1] : f32
inner.v: [4, 4] : f32
w: 2 : f32
8 : f32
```

## reduce

```python
m = [[1.0, 2.0], [3.0, 4.0]]
print(sum(m))
print(sum(m, 0))
print(sum(m, 1))
print(mean(m))
print(mean(m, 1))
```

Output:

```
10 : f32
[4, 6] : f32
[3, 7] : f32
2.5 : f32
[1.5, 3.5] : f32
```

## reduce_minmax

```python
m = [[1.0, 5.0], [7.0, 3.0]]
print(max(m))
print(max(m, 0))
print(max(m, 1))
print(min(m))
print(maximum([1.0, 4.0], [3.0, 2.0]))
print(minimum(2.0, [1.0, 3.0]))

function peak(x):
  max(x)

print(grad(peak, [1.0, 5.0, 3.0]))

function rowpeaks(x):
  sum(max(x, 1))

print(grad(rowpeaks, [[1.0, 5.0], [7.0, 3.0]]))
```

Output:

```
7 : f32
[7, 5] : f32
[5, 7] : f32
1 : f32
[3, 4] : f32
[1, 2] : f32
[0, 1, 0] : f32
[[0, 1], [1, 0]] : f32
```

## resize

```python
x = [[0.0, 1.0], [1.0, 0.0]]
print(resize(x, 4, 4))

function f(v):
  sum(resize(v, 4, 4))

print(grad(f, [[0.0, 1.0], [1.0, 0.0]]))

m = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
print(crop(m, 1, 0, 2, 2))

function c(v):
  sum(crop(v, 0, 1, 2, 2))

print(grad(c, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
```

Output:

```
[[0, 0.25, 0.75, 1], [0.25, 0.375, 0.625, 0.75], [0.75, 0.625, 0.375, 0.25], [1, 0.75, 0.25, 0]] : f32
[[4, 4], [4, 4]] : f32
[[4, 5], [7, 8]] : f32
[[0, 1, 1], [0, 1, 1]] : f32
```

## slicing

```python
d = [10.0, 20.0, 30.0, 40.0]
print(slice(d, 1.0, 2))
print(slice(d, 0.0, 3))
m = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
print(slice(m, 2.0, 1))
print(floor([1.7, -1.2, 3.0]))

function pick(x):
  sum(slice(x, 1.0, 2))

print(grad(pick, [10.0, 20.0, 30.0, 40.0]))

acc = 0.0
for i in 0..6:
  acc = acc + sum(slice([10.0, 20.0, 30.0], mod(i, 3.0), 1))
print(acc)
print(mod(-1.0, 3.0))
print(mod([5.0, 6.0, 7.0], 3.0))
```

Output:

```
[20, 30] : f32
[10, 20, 30] : f32
[[5, 6]] : f32
[1, -2, 3] : f32
[0, 1, 1, 0] : f32
120 : f32
2 : f32
[2, 0, 1] : f32
```

## text

```python
x = load("tests/cases/data/hello.txt")
print(x)
print(text(x))
save(x, "tests/cases/data/copy.txt")
y = load("tests/cases/data/copy.txt")
print(sum(where(y == x, 1.0, 0.0)))

print(bincount([0.0, 1.0, 1.0, 3.0, 1.0], 4))

counts = bincount(x, 256)
present = where(counts > 0.0, 1.0, 0.0)
rank = cumsum(present) - 1.0
chars = take(rank, x)
print(sum(present))
print(chars)

ids = tokenize("tests/cases/data/aab.txt", "tests/cases/data/tok.json")
print(ids)
print(detokenize(ids, "tests/cases/data/tok.json"))
```

Output:

```
[104, 101, 108, 108, 111, 32, 118, 101, 99, 116, 111, 114] : f32
hello vector
12 : f32
[1, 3, 0, 1] : f32
9 : f32
[3, 2, 4, 4, 5, 0, 8, 2, 1, 7, 5, 6] : f32
[4, 5] : f32
aab aab
```

## train

```python
function loss(w):
  ys = load("tests/cases/data/v.npy")
  d = w - ys
  mean(d * d)

w = f64([0.0, 0.0])
for step in 0..10:
  w = w - 1.0 * grad(loss, w)
print(w)
print(loss(w))
```

Output:

```
[3, 4] : f64
0 : f64
```

## transpose

```python
m = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
print(transpose(m))

function f(x):
  sum(transpose(x) * [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])

print(grad(f, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))

function t(r):
  transpose(r)

print(vmap(t, [[[1.0, 2.0]], [[3.0, 4.0]]]))
```

Output:

```
[[1, 4], [2, 5], [3, 6]] : f32
[[1, 2, 3], [4, 5, 6]] : f32
[[[1], [2]], [[3], [4]]] : f32
```

## vmap

```python
function double(x):
  x * 2.0

print(vmap(double, [1.0, 2.0, 3.0]))

function nsq(v):
  sum(v * v)

print(vmap(nsq, [[3.0, 4.0], [6.0, 8.0]]))

function dotp(a, b):
  sum(a * b)

print(vmap(dotp, [[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]))

function lin(x):
  matmul([[1.0, 0.0], [0.0, 2.0]], x)

print(vmap(lin, [[1.0, 1.0], [2.0, 3.0]]))

function dnsq(v):
  grad(nsq, v)

print(vmap(dnsq, [[3.0, 4.0], [1.0, 2.0]]))

function agg(m):
  sum(vmap(nsq, m))

print(grad(agg, [[1.0, 2.0], [3.0, 4.0]]))

function perl(x):
  sum(matmul([[1.0, 2.0], [3.0, 4.0]], x))

function aggl(m):
  sum(vmap(perl, m))

print(grad(aggl, [[1.0, 1.0], [1.0, 1.0]]))
```

Output:

```
[2, 4, 6] : f32
[25, 100] : f32
[17, 53] : f32
[[1, 2], [2, 6]] : f32
[[6, 8], [2, 4]] : f32
[[2, 4], [6, 8]] : f32
[[4, 6], [4, 6]] : f32
```

## vmap_nested

```python
function cell(c):
  c * c

function row(r):
  vmap(cell, r)

print(vmap(row, [[1.0, 2.0], [3.0, 4.0]]))

function rowsum(r):
  sum(vmap(cell, r))

print(vmap(rowsum, [[1.0, 2.0], [3.0, 4.0]]))

function aggn(m):
  sum(vmap(rowsum, m))

print(grad(aggn, [[1.0, 2.0], [3.0, 4.0]]))
```

Output:

```
[[1, 4], [9, 16]] : f32
[5, 25] : f32
[[2, 4], [6, 8]] : f32
```

## while

```python
x = 1.0
while x < 100.0:
  x = x * 2.0
print(x)

v = [3.0, 1.0, 4.0]
n = 0.0
while norm(v) > 0.1:
  v = v * 0.5
  n = n + 1.0
print(n)

y = 100.0
steps = 0.0
while abs(y - sqrt(2.0)) > 0.000001:
  y = 0.5 * (y + 2.0 / y)
  steps = steps + 1.0
print(y)
print(steps)

m = {a: 1.0, b: [1.0, 1.0]}
while m.a < 10.0:
  m = m * 2.0
print(m)
```

Output:

```
128 : f32
6 : f32
1.4142135 : f32
10 : f32
a: 16 : f32
b: [16, 16] : f32
```
