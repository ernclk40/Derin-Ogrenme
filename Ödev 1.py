X = [[0,0],[0,1],[1,0],[1,1]]
y = [0, 0, 0, 1]                # AND kapısı

w = [0.40, -0.52]               # elle seçtiğimiz başlangıç ağırlıkları
b = 0.66                        # elle seçtiğimiz bias
lr, epoch = 0.1, 0

while True:
    dw, db, hata = [0, 0], 0, 0
    for x, t in zip(X, y):
        o = 1 if w[0]*x[0] + w[1]*x[1] + b > 0 else 0
        e = t - o
        dw[0] += lr*e*x[0]; dw[1] += lr*e*x[1]; db += lr*e
        hata += abs(e)
    w[0] += dw[0]; w[1] += dw[1]; b += db   # epoch sonunda tek güncelleme
    epoch += 1
    if hata == 0:
        break

print("epoch:", epoch)
print("w:", [round(v, 2) for v in w], "b:", round(b, 2))
for x, t in zip(X, y):
    print(x, t, 1 if w[0]*x[0] + w[1]*x[1] + b > 0 else 0)