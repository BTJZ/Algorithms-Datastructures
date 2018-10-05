#include<bits/stdc++.h>

using namespace std;

const int maxn = 100000 + 5;

typedef long long ll;

int n, m;

ll inline sum(ll a, ll b) { return a + b; }

ll val[maxn * 4], tag[maxn * 4], num; int s, t;
void inline pushdown(int id, int l, int r, int mid) { 
	tag[id << 1] += tag[id], val[id << 1] += tag[id] * (mid - l + 1);
	tag[id << 1 | 1] += tag[id], val[id << 1 | 1] += tag[id] * (r - mid);
	tag[id] = 0; 
}
void Add(int id, int l, int r) {
	if(s <= l && t >= r) { tag[id] += num, val[id] += num * (r - l + 1); return; }
	int mid = l + r >> 1;
	pushdown(id, l, r, mid);
	if(s <= mid) Add(id << 1, l , mid);
	if(t > mid) Add(id << 1 | 1, mid + 1, r);
	val[id] = sum(val[id << 1], val[id << 1 | 1]);
}
ll Query(int id, int l, int r) {
	if(s <= l && t >= r) { return val[id]; }
	int mid = l + r >> 1; ll a, b = a = 0;
	pushdown(id, l, r, mid);
	if(s <= mid) a = Query(id << 1, l, mid);
	if(t > mid) b = Query(id << 1 | 1, mid + 1, r);
	return sum(a, b);
}

void Build(int id, int l, int r) {
	if(l == r) { cin >> val[id]; return; }
	int mid = l + r >> 1;
	Build(id << 1, l , mid);
	Build(id << 1 | 1, mid + 1, r);
	val[id] = sum(val[id << 1], val[id << 1 | 1]);
}

int main()
{
	cin >> n >> m;
	Build(1, 0, n - 1);
	int opt;
	while(m--) {
		cin >> opt >> s >> t;
		s -= 1, t -= 1;
		if(opt == 1) {
			cin >> num;
			Add(1, 0, n - 1);
		} else {
			cout << Query(1, 0, n - 1) << endl;
		}
	}
    return 0;
}
