##zkw(lbw)线段树
####------------by zzy
**lbw线段树是一种极其极其极其实用的**~~简单~~**数据结构**
####例题 
>1.[P3374 模板1](https://www.luogu.org/problemnew/show/P3374)
>2.[P3368 模板2](https://www.luogu.org/problemnew/show/P3368)
>3.[P3372 模板3](https://www.luogu.org/problemnew/show/P3372)
>4.[P3865 模板4](https://www.luogu.org/problemnew/show/P3865)
>5.[P3369 模板5](https://www.luogu.org/problemnew/show/P3369)

####核心代码
**1.单点修改，区间查询**
```cpp
#include <iostream>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <algorithm>
#define maxn 500005
using namespace std;

int n, m, N, tree[maxn << 2];

void build(){//建树
    for(int i = N - 1; i; --i)
        tree[i] = tree[i << 1] + tree[i << 1 | 1];
}

void update(int d, int k){//单点修改
    for(d += N; d; d >>= 1)
    	tree[d] += k;
}

int query(int l, int r){//区间查询
    int ans = 0;
    for(l += N - 1, r += N + 1; l ^ r ^ 1; l >>= 1, r >>= 1){
        if(~l & 1)
            ans += tree[l ^ 1];
        if(r & 1)
            ans += tree[r ^ 1];
    }
    return ans;
}

void init(){
    cin >> n >> m;
    for(N = 1; N < n + 2; N <<= 1);
    for(int i = N + 1; i <= N + n; ++i)
    	cin >> tree[i];
    build();
    return;
}

void solve(){
    int a, b, c;
    while(m--){
        scanf("%d %d %d", &a, &b, &c);
        if(a == 1)
            update(b, c);
        else
            cout << query(b, c) << endl;
    }
    return;
}

int main(){
    init();
    solve();
    return 0;
}
```
**2.区间修改，区间查询**
```cpp
#include <iostream>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <algorithm>
#define ll long long
#define maxn 100005
#define ent putchar('\n')
#define REP(i,x,y) for(register int i = x; i <= y; ++i)
using namespace std;

inline int read(){
    int x = 0, f = 1; char s = getchar();
    while(s < '0'||s > '9') {if(s == '-') f = -1; s = getchar();}
    while(s >= '0'&&s <= '9') {x = x * 10 + s -'0'; s = getchar();}
    return x * f;
}

template<class T> inline void print(T x){
    if(x < 0) {putchar('-'); x = -x;}
    if(x > 9) print(x / 10);
    putchar(x % 10 + '0');
}

int n, m, M;
ll ori[maxn], tree1[maxn << 2], tree2[maxn << 2];

void build(){//tree1为差分数组的树，tree2为tree1乘点编号
    REP(i, 1, n){
        tree1[i + M] = ori[i] - ori[i - 1];
        tree2[i + M] = i * tree1[i + M];
    }
    for(int i = M - 1; i; --i){
        tree1[i] = tree1[i << 1] + tree1[i << 1 | 1];
        tree2[i] = tree2[i << 1] + tree2[i << 1 | 1];
    }
}

inline void add(int d, int k){//差分数组树的单点修改
    int dk = d * k;
    for(d += M; d; d >>= 1){
        tree1[d] += k;
        tree2[d] += dk;
    }
}

inline void update(int l, int r, int k){//区间修改
    add(l, k); add(r + 1, -k);
}

inline ll que(int d){//前缀和查询
    ll ans1 = 0, ans2 = 0;
    int temp = d + M + 1;
    while(temp > 1){
        if(temp & 1){
            ans1 += tree1[temp ^ 1];
            ans2 += tree2[temp ^ 1];
        }
        temp >>= 1;
    }
    return (d + 1) * ans1 - ans2;//推导的数学公式
}

inline ll query(int l, int r){//区间查询
    return que(r) - que(l - 1);
}

void init(){
    n = read(); m = read();
    for(M = 1; M < n + 2; M <<= 1);
    REP(i, 1, n)
    	ori[i] = read();
    build();
    return;
}

void solve(){
    int a, b, c;
    while(m--){
    	a = read(); b = read(); c = read();
    	if(a == 1){
    		a = read();
    		update(b, c, a);
        }
        else{
    		print(query(b, c)); ent;
        }
    }
    return;
}

int main(){
    init();
    solve();
    return 0;
}
```
**3.区间修改，区间最大值查询**
```cpp
#include <iostream>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <algorithm>
#define maxn 100005
#define ent putchar('\n')
#define REP(i,x,y) for(int i = x; i <= y; ++i)
using namespace std;

inline int read(){
    int x = 0, f = 1; char s = getchar();
    while(s < '0'||s > '9') {if(s == '-') f = -1; s = getchar();}
    while(s >= '0'&&s <= '9') {x = x * 10 + s -'0'; s = getchar();}
    return x * f;
}

template<class T> inline void print(T x){
    if(x < 0) {putchar('-'); x = -x;}
    if(x > 9) print(x / 10);
    putchar(x % 10 + '0');
}


int n, m, M, ori[maxn], tree[maxn << 2];

void build(){//建差分树
    for(int i = M - 1; i; --i){
        tree[i] = max(tree[i << 1], tree[i << 1 | 1]);
        tree[i << 1] -= tree[i];
        tree[i << 1 | 1] -= tree[i];
    }
}

int query(int l, int r){//RMQ查询
    int ans, lans, rans;
    for(lans = tree[l += M], rans = tree[r += M]; l ^ r ^ 1 && l != r; l >>= 1, r >>= 1, lans += tree[l], rans += tree[r]){
        if(~l & 1)
            lans = max(lans, tree[l ^ 1]);
        if(r & 1)
            rans = max(rans, tree[r ^ 1]);
    }
    ans = max(lans, rans);
    while(l > 1)
        ans += tree[l >>= 1];
    return ans;
}

void init(){
    n = read(); m = read();
    for(M = 1; M < n + 2; M <<= 1);
    REP(i, 1, n)
    	tree[i + M] = read();
    build();
    return;
}

void solve(){
    int a, b;
    while(m--){
    	a = read(); b = read();
    	print(query(a, b)); ent;
    }
    return;
}

int main(){
    init();
    solve();
    return 0;
}
```
**4.代替平衡树**
```cpp
#include <iostream>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#define maxn 100005
#define INF 0x3f3f3f3f
#define ll long long
#define ent putchar('\n')
#define REP(i,x,y) for(register int i = x; i <= y; ++i)
#define rep(i,x,y) for(register int i = x; i >= y; --i)
using namespace std;

inline int read(){
    int x = 0, f = 1; char s = getchar();
    while(s < '0'||s > '9') {if(s == '-') f = -1; s = getchar();}
    while(s >= '0'&&s <= '9') {x = x * 10 + s -'0'; s = getchar();}
    return x * f;
}

template<class T> inline T print(T x){
    if(x < 0) {putchar('-'); x = -x;}
    if(x > 9) print(x / 10);
    putchar(x % 10 + '0');
}

int n, m;
int lg[maxn], MAX[20][maxn];

int query(int l, int r){
    int k = lg[r - l + 1] - 1;
    return max(MAX[k][l], MAX[k][r - (1 << k) + 1]);
}

inline void init(){
    n = read(), m = read();
    REP(i, 1, n) lg[i] = lg[i - 1] + (1 << lg[i - 1] == i);
    REP(i, 1, n) MAX[0][i] = read();
}

inline void solve(){
    rep(i, n, 1) for(register int j = 1; 1 << j <= n - i + 1; ++j)
      MAX[j][i] = max(MAX[j - 1][i], MAX[j - 1][i + (1 << (j - 1))]);
    int l, r;
    while(m--){
        l = read(), r = read();
        print(query(l, r)), ent;
    }
    return;
}

int main(){
    init();
    solve();
    return 0;
}
```
**4.代替平衡树**
```cpp
#include <iostream>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <algorithm>
#define ent putchar('\n')
#define maxn 100005
#define REP(i,x,y) for(int i = x; i <= y; ++i)
using namespace std;

inline int read(){ 
	int x = 0, f = 1; char s = getchar();
	while(s > '9' || s < '0') {if(s == '-') f = -1; s = getchar();}
	while(s <= '9' && s >= '0') {x = x * 10 + s - '0'; s = getchar();}
	return x * f;
}

inline void print(int x){//快速输出 
	if(x < 0) {putchar('-'); x = -x;}
	if(x > 9) print(x / 10);
	putchar(x % 10 + '0');
}

//总思想：树的每个叶节点存每个数的出现次数
//缺点：若数的范围过大需离散化，不可在线算 

int n, M, tree[maxn << 2], que[2][maxn], c[maxn];//que存询问信息，c是用来离散化的数组 

inline void update(int d, int k){//单点更新 
	for(d += M; d; d >>= 1)
		tree[d] += k;
}

inline int Rank(int d){//查询某个数的排名 
	int ans = 0;
	for(d += M; d; d >>= 1)//查询以d - 1为右端点的前缀和 
		if(d & 1)
			ans += tree[d ^ 1];
	return ans + 1;//答案为以d - 1为右端点的前缀和  + 1 
}

inline int query(int k){//查询以k为排名的数 
    int res = 1;//从根节点向下找 
	while(res < M){//找到叶子节点 
		if(tree[res << 1] >= k)//若左结点表示区间内的数不少于k，则以k为排名的数一定在左节点表示区间内 
			res <<= 1;
		else{
			k -= tree[res << 1];//否则，在右子节点内查询排名为(k - 左子节点表示区间内数的个数)的数 
			res = res << 1 | 1;
		}
	}
	return c[res - M];//返回离散化前的数 
}

inline int find_pre(int d){//寻找前驱(比它小的数中最大的数，题目保证此数存在)
	int temp;
	for(d += M; d; d >>= 1)//找到d前面的第一个区间内数总出现次数不为0的区间，即为temp
		if((d & 1) && tree[d ^ 1]){
			temp = d ^ 1;
			break;
		}
	while(temp < M){//找到这个区间最靠右(即最大)且出现次数不为0的数，即为d的前驱
		if(tree[temp << 1 | 1])
			temp = temp << 1 | 1;
		else
			temp <<= 1;
	}
	return c[temp - M];
}

inline int find_nxt(int d){//寻找后继(类似于求前驱) 
	int temp;
	for(d += M; d; d >>= 1)
		if((~d & 1)	&& tree[d ^ 1]){
			temp = d ^ 1;
			break;
		}
	while(temp < M){
		if(tree[temp << 1])
			temp <<= 1;
		else
			temp = temp << 1 | 1;
	}
	return c[temp - M];
}

void init(){
	int len = 0;
	n = read();
	REP(i, 1, n){
		que[0][i] = read(); que[1][i] = read();//读入询问 
		if(que[0][i] < 2 || que[0][i] > 4)//1，5，6操作的数需离散化；2，3操作的数在1操作中一定出现 
			c[++len] = que[1][i];
	}
	sort(c + 1, c + len + 1);
	len = unique(c + 1, c + len + 1) - c - 1; 
	for(M = 1; M < len + 2; M <<= 1);//用不同数的个数(len)求M
	REP(i, 1, n)
		if(que[0][i] != 4)
			que[1][i] = lower_bound(c + 1, c + len + 1, que[1][i]) - c;//将除4操作外的数离散为它们在c数组中的下标(保证 < n) 
	return;
}

void solve(){
	REP(i, 1, n){
		if(que[0][i] == 1)//加数即为将其出现次数 + 1 
			update(que[1][i], 1);
		else if(que[0][i] == 2)//删数即为将其出现次数 - 1 
			update(que[1][i], -1);
		else if(que[0][i] == 3){
			print(Rank(que[1][i])); ent;
		}
		else if(que[0][i] == 4){
			print(query(que[1][i])); ent;
		}
		else if(que[0][i] == 5){
			print(find_pre(que[1][i])); ent;
		}
		else{
			print(find_nxt(que[1][i])); ent;
		}
	}
	return;
}

int main(){
	init();
	solve();
	return 0;
}
```
**5.区间修改RMQ**
由于太毒瘤已被屏蔽