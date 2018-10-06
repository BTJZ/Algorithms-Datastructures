#include<bits/stdc++.h>

using namespace std;

const int maxn = 100000 + 5;
const int maxm = 100000 + 5;

int n, m, r, p;
int head[maxn], to[maxn * 2], nxt[maxn * 2], w[maxn], cnt;
void addEdge(int u, int v) {
	to[++cnt] = v, nxt[cnt] = head[u], head[u] = cnt;
	to[++cnt] = u, nxt[cnt] = head[v], head[v] = cnt;
}

int fa[maxn], son[maxn], top[maxn], size[maxn], deep[maxn], dfn[maxn], ogn[maxn], dfs_clock;
void dfs1(int u, int pa) 
{
	deep[u] = deep[pa] + 1;
	fa[u] = pa;
	size[u] = 1;
	for(int e = head[u]; e; e = nxt[e]) {
		int v = to[e];
		if(v == pa) continue;
		dfs1(v, u);
		size[u] += size[v];
		if(size[v] > size[son[u]]) son[u] = v;
	}
}
void dfs2(int u, int pa) 
{
	dfn[u] = ++dfs_clock, ogn[dfs_clock] = u;
	if(son[u]) {
		top[son[u]] = top[u];
		dfs2(son[u], u);
	}
	for(int e = head[u]; e; e = nxt[e]) {
		int v = to[e];
		if(v == pa || v == son[u]) continue;
		top[v] = v;
		dfs2(v, u);
	}
}
int val[maxn * 4], tag[maxn * 4], s, t, num;
void build(int id, int l, int r) 
{
	if(l == r) { val[id] = w[ogn[l]] % p; return; }
	int mid = l + r >> 1;
	build(id << 1, l, mid);
	build(id << 1 | 1, mid + 1, r);
	val[id] = (val[id << 1] + val[id << 1 | 1]) % p;
}
void inline pushdown(int id, int l, int r, int mid) {
	tag[id << 1] += tag[id], tag[id << 1 | 1] += tag[id];
	val[id << 1] = (val[id << 1] + tag[id] * (mid - l + 1)) % p, val[id << 1 | 1] = (val[id << 1 | 1] + tag[id] * (r - mid)) % p;
	tag[id] = 0;
}
void radd(int id, int l, int r) 
{
	if(s <= l && t >= r) { val[id] = (val[id] + num * (r - l + 1)) % p, tag[id] += num; return; }
	int mid = l + r >> 1;
	pushdown(id, l, r, mid);
	if(s <= mid) radd(id << 1, l, mid);
	if(t > mid) radd(id << 1 | 1, mid + 1, r);
	val[id] = (val[id << 1] + val[id << 1 | 1]) % p;
}
int query(int id, int l, int r) 
{
	if(s <= l && t >= r) return val[id];
	int mid = l + r >> 1;
	pushdown(id, l, r, mid);
	int ans = 0;
	if(s <= mid) ans = (ans + query(id << 1, l, mid)) % p;
	if(t > mid) ans = (ans + query(id << 1 | 1, mid + 1, r)) % p;
	return ans;
}

void Init()
{
	scanf("%d %d %d %d", &n, &m, &r, &p);
	for(int i = 1; i <= n; ++i) scanf("%d", w + i);
	int u, v;
	for(int i = 1; i < n; ++i) { 
		scanf("%d %d", &u, &v);
		addEdge(u, v);
	}
	dfs1(r, 0);
	top[r] = r;
	dfs2(r, 0);
	build(1, 1, n);
}

int opt, x, y, z;

void inline addWay()
{
	while(top[x] != top[y]) 
	{
		if(deep[top[x]] > deep[top[y]]) {
			s = dfn[top[x]], t = dfn[x], num = z;
			radd(1, 1, n);
			x = fa[top[x]];
		} else {
			s = dfn[top[y]], t = dfn[y], num = z;
			radd(1, 1, n);
			y = fa[top[y]];
		}
	}
	s = dfn[x], t = dfn[y], num = z; if(s > t) swap(s, t);
 	radd(1, 1, n);
}
int inline queryWay()
{
	int ans = 0;
	while(top[x] != top[y]) 
	{
		if(deep[top[x]] > deep[top[y]]) {
			s = dfn[top[x]], t = dfn[x];
			ans = (ans + query(1, 1, n)) % p;
			x = fa[top[x]];
		} else {
			s = dfn[top[y]], t = dfn[y];
			ans = (ans + query(1, 1, n)) % p;
			y = fa[top[y]];
		}
	}
	s = dfn[x], t = dfn[y]; if(s > t) swap(s, t);
 	return (ans + query(1, 1, n)) % p;
}
void inline addTree()
{
	s = dfn[x], t = s + size[x] - 1, num = z;
	radd(1, 1, n);
}
int inline queryTree()
{
	s = dfn[x], t = s + size[x] - 1;
	return query(1, 1, n);
}

void Solve()
{
	while(m--) 
	{
		scanf("%d", &opt);
		switch(opt) 
		{
		case 1:
			scanf("%d %d %d", &x, &y, &z);
			addWay();
			break;
		case 2:
			scanf("%d %d", &x, &y);
			printf("%d\n", queryWay());
			break;
		case 3:
			scanf("%d %d", &x, &z);
			addTree();
			break;
		case 4:
			scanf("%d", &x);
			printf("%d\n", queryTree());
			break;
		default:
			printf("WTF?\n");
		}
	}
}

int main()
{
	Init();
	Solve();
	return 0;
}
