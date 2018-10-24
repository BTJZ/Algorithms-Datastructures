# 目录

[TOC]

# 基础算法
## 排序
### 归并排序
```cpp
int n,a[1100000],c[1100000];//a为待排序数组，c为中间数组，不要忘记开两倍
long long ans;              //求逆序对数
void gb(int x,int y)        //此处为归并排序
{
    if(x==y)  return;       //当区间减为1时，返回
    int mid=(x+y)/2;        //将区间二分
    gb(x,mid);              //左右分别进行归并排序(分治)
    gb(mid+1,y); 
    int i=x,j=mid+1,k=x;    //此时左右区间分别有序，i，j为左右区间首，k为此段区间首
    while(i<=mid&&j<=y)     //当左右区间都未放进中间数组时
    {
        if(a[i]<=a[j])          
        {c[k]=a[i];k++;i++;}     //谁小谁先放进中间数组，指针后移一位
        else
        {c[k]=a[j];k++;j++;
        ans+=mid-i+1;} 			  //此处为用归并排序求逆序对数，详见逆序对定义，注意：“天对天多一天”	   
    } 	 
    while(i<=mid)                
    {c[k]=a[i];k++;i++;}         //如果左或右仍有未放入的，一定大于之前放入的
    while(j<=y)                  //直接放进中间数组
    {c[k]=a[j];k++;j++;}
    for(i=x;i<=y;i++)
    a[i]=c[i];                   //中间数组压入原数组
}
```
### 桶排
```cpp
int n,tp[10000];                          //桶排数组,需初始化 
int main()
{
    int i,j;
    cin>>n;                 //读入n
    for(i=1;i<=n;i++)       //读入n个数据
    { 
        cin>>j;   
        tp[j]++;            //用数组下标进行排序 
    }                                                  //桶排：将n个输入元素分配到这些桶中，对桶中元素进行排序                                            
    for(i=1;i<=n;i++)                                  //时间复杂度为O(N)，空间复杂度为O(N)  
    {                                                  //特点：快，但空间浪费严重，要求数据连续并小等n
        while(tp[i])
        {
            cout<<i<<" ";
            tp[i]--;        //处理重复数据
        }
    }
    return 0;               //若有负数，数组开两倍n处理即可
} 
```
## 分治
### 二分法
### 三分法
```cpp
double inline search(double l, double r) 
{
	double mid1, mid2;
	while(r - l > eps) {
		mid1 = (r - l) / 3 + l;
		mid2 = r - (r - l) / 3;
		if(f(mid1) > f(mid2)) r = mid2;
		else l = mid1;
	}
	return l;
}
```
### 二分答案
```cpp
int inline Solve(int l, int r)
{
    int mid;
    while(l < r) {
        mid = (l + r) >> 1;
        if(judge(mid)) r = mid;
        else l = mid + 1;
    }
    return l;
}
```
### 快速冪

```cpp
ll power(ll a, ll b, int p) {
    ll ans = 1;
    for(; b; b >>= 1, a = a * a % p)
        if(b & 1) ans = ans * a % p;
    return ans;
}
```

# 基础数据结构
## 堆 
**------------by zzy**

堆的实质是一个完全二叉树，其父节点比两个子节点都大（大根堆）或小（小根堆）

其堆顶元素必为最大值或最小值

### 例题

>[1.P3378 模板题]()
>[2.P1168 中位数]()
>[3.P2672 推销员]()

下面是一个大根堆

```cpp
int num = 0, a[maxn];
void putnum(int x){//将数x放入堆中
    a[++num] = x;
    int now = num, nxt = num >> 1;//now为新添节点，nxt为父节点
    while(nxt){
        if(a[now] >= a[nxt])//交换直到当前节点小于父节点
	        break;
        swap(a[now], a[nxt]);//若当前节点小于父节点，交换
        now = nxt; nxt = now >> 1;
    }
}

int gettop(){//返回最大值
    return a[1];
}

void poptop(){//删除最大值
    swap(a[1], a[num--]);//直接删除堆顶数，并将随便一个数（这里是最后一个数）移至堆顶
    int now = 1, nxt = now << 1;//now为转移的节点
    while(nxt <= num){
        if(nxt != num && a[nxt] > a[nxt + 1])//nxt为子节点中较大的
	        nxt++;
        if(a[now] <= a[nxt])//交换直到当前节点大于子节点
	        break;
        swap(a[now], a[nxt]);////若有子节点大于当前节点，交换较大子节点和当前节点
        now = nxt; nxt = now << 1;
    }
}
```
当然你也可以用优先队列来做
```cpp
priority_queue<int , vector<int> , greater<int> > q;
```
ps:手敲的堆要比优先队列快好多

## 前缀和

### 例题

> [冰河峡谷（QZA）]()

```cpp
void a()
 {
   int a[maxn],n,b[maxn];
   n=re();
   f(i,1,n) a[i]=re();
   f(i,1,n) b[i]+=b[i-1];
   f(i,1,n) b[i]-=b[i-1];
}
```

# 搜索
## DFS深度优先搜索
**--------by chl**

### 例题：
> [luogu P1219 八皇后](https://www.luogu.org/problemnew/show/P1219)
> [luogu P3958 奶酪](https://www.luogu.org/problemnew/show/P3958)

~~事实上如果策略得当很多题都可以用DFS~~

算法概念：深度优先搜索（DFS），本质上是一种暴力枚举算法，通过生成所有可能的结果，并验证结果的正确性，来找到正解。

剪枝：一种优化策略，可以显著减少无效枚举的情况，加快枚举速度。（~~既然zzy说不用写那就不往下写了0.0，例题的题解十分详细~~）

以下为DFS标准代码：

```cpp
#include<cstdio>
using namespace std;

void DFS(int x,int depth)
{
	if(终止条件) return ;
    //其实这里最好可以加记忆化，但这不是我负责的范围，所以选择交给zzy
    do something;
    dfs(x+...,depth+1);
}


int main(void)
{
	......
    dfs(初始状态);
    输出可行状态或最优状态
    ......
    return 0;
}


```
## BFS
## 记忆化搜索
**------------by zzy**

### 例题 
>1.[P3183 [HAOI2016]食物链](https://www.luogu.org/problemnew/show/P3183)
>2.[P1879 [USACO06NOV]玉米田Corn Fields](https://www.luogu.org/problemnew/show/P1879)

以下为核心代码
```cpp
int sum[maxn];
dfs(int now){
	if(sum[now])
		return sum[now];
	int ans = 0;
	//一些操作
	return sum[now] = ans;
}
```
玉米田的记忆化搜索解法
```cpp
int m, n, ans, MAX;
int line[maxn];
int cnt, num[10005];
int res[15][10005];

int dfs(int step, int last){
    if(res[step][last]) return res[step][last];
    if(step == m + 1)
        return 1;
    REP(i, 1, cnt){
        if(num[i] & last || num[i] & line[step])
            continue;
        res[step][last] += dfs(step + 1, num[i]);
    }
    return res[step][last];
}

void init(){
    int t;
    scanf("%d %d", &m, &n);
    REP(i, 1, m)//含义为for(int i = 1; i <= m; ++i)
        REP(j, 1, n){
            scanf("%d", &t);
            line[i] <<= 1;
            line[i] += t ^ 1;
        }
    MAX = (1 << n) - 1;
    REP(i, 0, MAX)
        if((i & (i << 1)) == 0)
            num[++cnt] = i;
}

void solve(){
    REP(i, 1, cnt)
        if((num[i] & line[1]) == 0)
            ans += dfs(2, num[i]);
    printf("%d", ans % 100000000);
}

int main(){
    init();
    solve();
    return 0;
}
```
## 迭代加深
## 最优性/可行性剪枝

# 动态规划
## 最长上升子序列
## 背包
### 0/1背包
```cpp
int sumv，num，v[104]，w[104]，f[1004];
int main()
{
cin>>sumv>>num; for(int i=1;i<=num;i++) cin>>w[i]>>v[i];//输入数据
for(int i=1;i<=num;i++) for(int j=sumv;j>=1;j--)//注意：内层循环倒序
if(j>=w[i]) f[j]=max(f[j-w[i]]+v[i]，f[j]);//01背包状态转移
cout<<f[sumv];//输出数据
return 0;
}
```
### 完全背包
给定N件物品，每件物品的重量为wi，价值为vi，(1<=i<=n)，且每件物品都有无数件。现有一个最大载重为M的包，求在不超出包的最大载重的情况下，可以装下的物品总价值的最大值。

同样的我们可以考虑子问题。用fi，j表示在只有i个物品，总容量为j的情况下该问题的最优解。我们考虑第+1个物品，可以选择将其放进背包（如果放进的话是多少件）或者不放进背包。不妨将不放进背包视为放进背包的一种特殊情况（放进0件）。那么则有：

```cpp
//初始化数组↓，这一步非常重要！！！切勿遗漏 
for (int i = 0; i <= m; i++)f[0][i] = 0; 
for (int i = 1; i <= n; i++) 
for (int j = 0; j <= m; j++) { 
f[i][j] = 0; 
for (int k = 0; k <= j / w[i]; k++) 
f[i][j] = max(f[i][j]， f[i - 1][j - k * w[i]] + k * v[i]); //状态转移
}//f[n][m]即为结果
```

```cpp
代码优化：
for(int i=0;i<=m;i++) f[i]=0; 
for(int i=1;i<=n;i++) 
for(int j=0;j<=m;j++) //注意：此处是正循环 
if(j>=w[i]) f[j]=max(f[j]，f[j-w[i]]+v[i]); //状态转移
```
### 多重背包
给定N件物品，每件物品的重量为wi，价值为vi，(1<=i<=N)，每件物品有ni件。现有一个最大载重为M的包，求在不超出包的最大载重的情况下，可以装下的物品总价值的最大值。
这个问题高度类似于完全背包问题，只要加上个数的限制即可，状态转移方程为：

### 部分背包
### 分组背包
给定N件物品，每件物品的重量为wi，价格为mi，价值为vi，(1<=i<=N)，且均有无穷多件。现有一个最大载重为M的包和E的钱，求在不超出包的最大载重的情况下，用仅有的钱可以买下并装下的物品总价值的最大值。
二维或者多维条件的背包问题，均只需要在转移方程上面增加一维即可。例如在这个问题中，只需要在原有的完全背包问题上加上一维即可。状态转移方程为：
### 附带背包
## 状态压缩
## 树上DP
## 区间DP（环形DP）

# 贪心

# 图论
## 存图
### 临接矩阵
```cpp
int mat[2233][2233],n,m;//mat邻接矩阵

	cin >> n >> m;
	for (int a=1;a<=m;a++)//顺次读入边
	{
		int s,e,d;
		cin >> s >> e >> d;
		mat[s][e]=d;//表示从s到e的边权值为d
		mat[e][s]=d;//双向边
}
```
### 链式前向星
```cpp
int first[100010],en;//first[i]表示以i为起点的第一条边的编号
struct edge
{
    int to,w,nxt=0;//next会重名 	   
}ed[200010];//结构体里面放边的信息，to为终点，w为权值，nxt为下条边的编号
void add(int s,int e,int d)
{
    en++;                    //en为边数，即第几条边
    ed[en].nxt=first[s];     //从上往下加边，当前边的下一条为原来的第一条
    ed[en].to=e;             //终点
    ed[en].w=d;              //权值
    first[s]=en; 	         //第一条边为当前边
}//加边 
add(s,e,d);
add(e,s,d);//双向边需加两次
```
## 最短路
### Dijkstra
**--------------by ZHX**


直接贴代码
```cpp
int dist[maxn];
bool use[maxn];

struct rec
{
	int p,dist;
	rec(){}  //构造函数初始化大于一必须加 
	rec(int a,int b){p=a,dist=b;}
};

bool operator<(const rec &a,const rec &b)  //重载小于号 
{
	return a.dist>b.dist;
}

priority_queue<rec> heap;

void dijkstra_heap()
{
	memset(dist,0x3f,sizeof(dist));   //0x7f 会爆int 
	dist[1]=0;
	for (int a=1;a<=n;a++)
		heap.push(rec(a,dist[a]));
	
	for (int a=1;a<=n;a++)
	{
		while (use[heap.top().p])  //防止重复多个弹出同一点 
			heap.pop();
		rec now = heap.top();
		heap.pop();
		int p=now.p;
		use[p]=true;
		
		for (int i=first[p];i;i=ed[i].next)
			if (dist[p]+ed[i].changdu < dist[ed[i].zhongdian])
			{
				dist[ed[i].zhongdian] = dist[p]+ed[i].changdu;
				heap.push(rec(ed[i].zhongdian,dist[ed[i].zhongdian]));
			}
	}
}
```

### SPFA

syc 已经死了的spfa

#### 例题

> [1.洛谷 P3371]()
> [2.T49370]()

谁说是spfa死了的

```cpp
void spfa(){
	for(int i = 1 ; i <= n ; i++) d[i] = inf;
	d[start] = 0;
	Q.push(start);
	vis[start] = 1;
	while(!Q.empty()){
		int u = Q.front();
		Q.pop();
		vis[u] = 0;
		for(int i = head[x] ; i ; i = e[i].nxt){
			if(d[u]+e[i].val<d[e[i].to]){
				d[e[i].to] = d[u] + e[i].val;
				if(!vis[e[i].to]){
					Q.push(e[i].to);
					vis[e[i].to] = 1;
				}
			}
		}
	}
}
```
### Floyd
```cpp
void floyd(){
	for(int k = 1 ; k <= n ; k++)
		for(int i = 1 ; i <= n ; i++)
			for(int j = 1 ; j <= n ; j++)
				if(map[i][k]+map[k][j]<map[i][j])
				map[i][j] = map[i][k] + map[k][j];
}
```
## 最小生成树
### Kruskal 算法
**-----by CHL**

#### 例题 

>[1.luogu P3366 【模板】最小生成树 ](https://www.luogu.org/problemnew/show/P3366)
>[2.luogu P1967  货车运输](https://www.luogu.org/problemnew/show/P1967)
>[3.luogu P2504  聪明的猴子](https://www.luogu.org/problemnew/show/P12504)


生成树： 一张具有n个点，n-1条边的联通图
最小生成树：即为最小权重生成树，指总边权最小的生成树

生成最小生成树的算法：Kruskal 算法 && Prim 算法

两种算法的比较：在稠密图中Prim算法较优，在稀疏图中Kruskal算法较优

Kruskal算法

1.算法思想：贪心

2.算法基础： sort&&并查集

3.算法步骤：

（1）存入这张图

（2）按边权从小到大排序所有边

（3）从小到大地填边，把已填过的边合并为一个集合

贴代码

```cpp
#include<cstdio>
#include<algorithm>
#include<iostream>
const int maxn=200005;
int m,n;
int fa[maxn],ans,sum;    //ans存最小生成树总权值,sum存已加入的边数

struct Edge
{
	int u,v,w;
}e[maxn];

bool cmp(Edge a,Edge b)
{
	return a.w<b.w;   // 按照边权排序
}
int find(int v)
{
	if(v==fa[v]) return v;
	else return fa[v]=find(fa[v]);   // 并查集不再赘述	 
}

void Kruskal()
{
	for(int i=1;i<=m;i++)  //枚举所有边
    {
    	if(find(e[i].u)!=find(e[i].v)) //如果起终点未联通
        {
        	fa[find(e[i].u)]=find(e[i].v);    //加入一个集合中
            ans+=e[i].w;
            sum++;
		}
	}
}

int main(void)
{
	
	scanf("%d%d",&n,&m);  //读入 n个顶点 m条边
    for(int i=1;i<=m;i++)
    {
    	int u,v,w;
        scanf("%d%d%d",&u,&v,&w);
        e[i].u=u;e[i].v=v;e[i].w=w;
	}
	for(int i=1;i<=m;i++) fa[i]=i;
    std::sort(e+1,e+m+1,cmp);
    Kruskal();
    if(sum<n-1) printf("并不能生成生成树啦0.0");  //原因是给出的图并不是连通图
    else printf("%d\n",ans); //输出最小生成树总权值
    return 0;
}
```


### Kruskal重构树
## 最小生成树
### Prim算法
**--------------by ZHX**


还是直接贴代码
```cpp
int dist[maxn];
bool use[maxn];

struct rec
{
	int p,dist;
	rec(){}
	rec(int a,int b){p=a,dist=b;}
};

bool operator<(const rec &a,const rec &b)
{
	return a.dist>b.dist;
}

priority_queue<rec> heap;

void prim_heap()
{
	memset(dist,0x3f,sizeof(dist));
	dist[1]=0;
	for (int a=1;a<=n;a++)
		heap.push(rec(a,dist[a]));
	
	for (int a=1;a<=n;a++)
	{
		while (use[heap.top().p])
			heap.pop();
		rec now = heap.top();
		heap.pop();
		int p=now.p;
		use[p]=true;
		
		for (int i=first[p];i;i=ed[i].next)
			if (ed[i].changdu < dist[ed[i].zhongdian])
			{
				dist[ed[i].zhongdian] = ed[i].changdu;
				heap.push(rec(ed[i].zhongdian,dist[ed[i].zhongdian]));
			}
	}
}
```
## 二分图
## 树链剖分
```cpp
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
```
## LCA

### 例题

> [1.Luogu P3379]()

### RMQ
### 树上倍增

syc LCA

```cpp
int lca(int x,int y){
    int mmin = inf;
    if(deep[x]<deep[y]) swap(x,y);
    while(deep[x]>deep[y]){
        mmin = min(mmin,lca2[ lg[ deep[x] - deep[y] ] - 1][x]);
        x = lca1[ lg[ deep[x] - deep[y] ] - 1][x];
    } 
    if(x==y){
        return mmin;
    }
    for(int i = lg[ deep[x] ] ; i >=0 ; i--) if(lca1[i][x]!=lca1[i][y]) {
        mmin = min(mmin,lca2[i][x]);
        x = lca1[i][x];
        mmin = min(mmin,lca2[i][y]);
        y = lca1[i][y];
    }
    mmin = min(mmin,lca2[0][x]);
    mmin = min(mmin,lca2[0][y]);
    return mmin;
}
```
### Tarjan
## 有向图强联通分量

### 例题

> [上白泽慧音]()
>
> [抢掠计划]()
>
> [受欢迎的牛]()

```cpp
void tarjan(int t)
 {
    syc++;
    dfn[t]=syc,low[t]=syc;
    stack[++size]=t,instack[t]=1;
    for(int i=head[t];i;i=way[i].fa)
     {
        if(!dfn[way[i].ed]) 
        {
           tarjan(way[i].ed);
           low[t]=min(low[t],low[way[i].ed]);
        }
        if(instack[way[i].ed])
         low[t]=min(low[t],low[way[i].ed]); 
     }
    if(dfn[t]==low[t])
     {
        cnt++;
        while(stack[size]!=t)
        {
            belong[stack[size]]=cnt;
            instack[stack[size]]=0;
            size--;
        } 
        belong[stack[size]]=cnt;
        instack[stack[size]]=0;
        size--;
     } 
}
```
## 拓扑排序
**------------by zzy**

### 例题 
>1.[P1038 神经网络](https://www.luogu.org/problemnew/show/P1038)
>2.[P2661 信息传递](https://www.luogu.org/problemnew/show/P2661)
>3.[zzy loves SB（神犇）](https://www.luogu.org/problemnew/show/T46473)

以下为核心代码
```cpp
int in[maxn];//点的入度
queue<int> q;
void toposort(){
    for(int i = 1; i <= n; ++i)
	    if(!in[i])
		    q.push(i);//入度为0的点入队
    while(!q.empty()){
        int temp = q.front(); q.pop();
        //一些操作
        for(int i = head[temp]; i; i = edge[i].nxt)
	        int v = edge[i].to;
	        if(--in[v] == 0)//删除从此点出发的边，删后边终点入度为0的入队
		        q.push(v);
    }
}
```

# 高级数据结构
## 蛤希
## 树状数组
**------------by zzy**

### 例题 
>1.[P3374 模板1](https://www.luogu.org/problemnew/show/P3374)
>2.[P3368 模板2](https://www.luogu.org/problemnew/show/P3368)
>3.[P1908 逆序对](https://www.luogu.org/problemnew/show/P1908)

以下为核心代码
```cpp
int n, m;
int tree[maxn];//树状数组

int lowbit(int x){//玄学lowbit操作
    return x & (-x);
}

void update(int d, int k){//将d节点增加k
    while(d <= n){
        tree[d] += k;
        d += lowbit(d);
    }
}

int que(int d){//查询前缀和
    int ans = 0;
    while(d){
        ans += tree[d];
        d -= lowbit(d);
    }
    return ans;
}

int query(int a, int b){//区间查询，
    return que(b) - que(a - 1);
}

void init(){//初始化
    n = read(), m = read();
    for(int i = 1; i <= n; ++i){
	    int k = read(); update(i, k);
	}
}
```
## 线段树
```cpp
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
```
## ZKW线段树

**------------by zzy**

**zkw线段树是一种极其极其极其实用的**~~简单~~**数据结构**

### 例题 
>1.[P3374 模板1](https://www.luogu.org/problemnew/show/P3374)
>2.[P3368 模板2](https://www.luogu.org/problemnew/show/P3368)
>3.[P3372 模板3](https://www.luogu.org/problemnew/show/P3372)
>4.[P3865 模板4](https://www.luogu.org/problemnew/show/P3865)
>5.[P3369 模板5](https://www.luogu.org/problemnew/show/P3369)

### 核心代码
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
**3.区间最大值查询**
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
## 权值线段树
## 并查集
 **------by CHL**

some very easy 例题

>[luogu P3367  【模板】并查集 ](https://www.luogu.org/problemnew/show/P3367)
>[luogu P1551   亲戚](https://www.luogu.org/problemnew/show/P1551)
>[luogu P3958   奶酪](https://www.luogu.org/problemnew/show/P3958)
>[luogu P1525   关押罪犯](https://www.luogu.org/problemnew/show/P1525)
>[luogu P1196   银河英雄传说  (@达哥）](https://www.luogu.org/problemnew/show/P1196)  
>[luogu P2024   食物链](https://www.luogu.org/problemnew/show/P2024)

直接贴代码

```cpp
#include<cstdio>
#include<iostream>
using namespace std;
int fa[maxn]; //存储每个节点的根节点(父亲)
//初始化  ==> for(int i=1;i<=maxn;i++) fa[i]=i; 即初始时自己的根节点均为自己 

int find(int v) //查询根节点 
{
	if(v==fa[v]) return v; //如果根节点为自己直接返回 
	else
	{
		fa[v]=find(fa[v]) //递归地向上找出根节点并把当前
  			             //节点的父亲节点标记为根节点 
        return fa[v];
	}
} /*
  	事实上，我们并不关心一个节点的父亲节点是谁，只需要关心它是否在一个集合中
	==> 路径压缩优化   
*/

int find(int v)
{
	if(v==fa[v]) return v;
	return fa[v]=find(fa[v]);
}

int find(int v)
{
	return v==fa[v]? v:fa[v]=find(fa[v]);
}

//以上只是不同写法0.0

void merge(int x,int y) //合并操作
{
	int t1=find(x) ,t2=find(y); //找到需要合并的两个节点的根节点 
	if(t1!=t2) // 若不在一个集合中 
			   fa[t2]=t1;  //合并
	return ;   
}
/*
  	关于某种名为启发式合并（按秩合并）的正统优化
	  ==> 钟(c)长(h)者(l)也不会 
	  于是若有被卡合并操作的可能(NOIP一般不会有）
	  ==> rand()合并大法好 
*/

void merge(int x,int y) 
{
	int t1=find(x),t2=find(y);
	if (t1!=t2) 
	{
		if (rand()%2) fa[t1]=t2; //玄学防卡 
		else fa[t2]=t1;
	}
}


int main(void)
{
	for(int i=1;i<=maxn;i++) fa[i]=i;
	char a[233];
	for(int i=1;i<=Q;i++) //Q表示Q次询问
	{
		cin>>a+1;
		if(a[1]=="chaxun") 
		{
			int x,y; //读入需要查询的节点 
			scanf("%d%d",&x,&y);
			if(find(x)==find(y)) puts("in the same jihe");  
		}
		if(a[1]=="hebing")
		{
			int x,y; //读入需要合并的节点 
			scanf("%d%d",&x,&y);
			merge(x,y);
		}
	} 
	return 0;
}
```
## 带权并查集

### 例题

> [银河英雄传说]()

```cpp
 int find(int x)
  {
 	if(fa[x]!=x)
    {
 	int kkk=fa[x];
 	fa[x]=find(fa[x]);
 	rank[x]=rank[x]+rank[kkk]; 
    }
 	return fa[x];
  } 

void merge(int x,int y)
{
  int x1=find(x);
  int y1=find(y);
  if(x1!=y1)
  fa[x1]=y1; 
  rank[x1]=你要的rank值； 
} 
```
## ST表

syc 区间RMQ最大值

### 例题

> [Luogu P3865]()

提示:
数据过大不要用cin
cout 也不能用

```cpp

void make(){
	bit[0] = 1; for(int i = 1 ; i <= 24 ; i++) bit[i] = bit[i-1] * 2;
	for(int i = 2 ; i <= n ; i++) lg[i] = lg[i>>1] + 1;
	for(int i = 1 ; i <= lg[n] ; i++)
		for(int j = 1 ; j <= n ;j++)
			st[i][j] = max(st[i-1][j],st[i-1][j+bit[i-1]]);
}

void find(){
	int l,r;
	scanf("%d%d",&l,&r);
	int k = lg[r-l+1];
	printf("%d\n",max(st[k][l],st[k][r-bit[k]+1]));
}
```

# 数学

## 唯一分解定理
## gcd
### exgcd
```cpp
void ex_gcd(int a,int b,int &x,int &y){
	if(b==0){
		c = a;
		x = 1;
		y = 0;
		return ;
	}
	ex_gcd(b,a%b,y,x);
	y -= x*(a/b);
}
```
### 逆元

syc 线性求逆元  高效求单个数逆元使用Ex_gcd 

>[P3811]()

```cpp
long long  n,p;

long long a[3000000 + 5];

int main(){
    scanf("%lld%lld",&n,&p);
    a[1] = 1;
    printf("%lld\n",a[1]);
    for(int i = 2 ; i <= n ; i++){
        a[i] = -(p/i)*a[p%i]%p;
        a[i] = (a[i]+p)%p;
        printf("%lld\n",a[i]);
    }
    return 0;
}
```
## 线性筛

syc 欧拉筛法 
>[P3383]() 

```cpp
flag[1] = 1;
for(int i = 2 ; i <= n ; i++){
	if(!flag[i]) prime[++cnt] = i;
    for(int j = 1 ; j <= cnt && prime[j]*i <= n ; j++){
        flag[i*prime[j]] = 1;
        if(i%prime[j]==0) break;
    }
}
```
## 中国剩余定理
## 高斯消元

```cpp
int n;
double A[maxn + 1][maxn] = {0};

void Gauss()
{
    int i, j, k, r;
    for(i = 0; i < n; ++i)
    {
        r = i;
        for(j = i + 1; j < n; ++j)
            if(fabs(A[j][i]) > fabs(A[r][i])) r = j;
        if(r != i) for(j = 0; j <= n; ++j) swap(A[r][j], A[i][j]);
        
        for(j = n; j >= i; --j) {
            for(k = i + 1; k < n; ++k)
                A[k][j] -= A[k][i] / A[i][i] * A[i][j];
        }
    }
    for(i = n - 1; i >= 0; --i) {
        for(j = i + 1; j < n; ++j)
            A[i][n] -= A[j][n] * A[i][j];
        A[i][n] /= A[i][i];
    }
}
```

另外，需要判断无解的情况

```cpp
for(int i = 0; i < n; ++i) if(A[i][n] != A[i][n]) {
    cout << "No Solution\n";
    return;
}
```

## 矩阵乘法/矩阵快速冪

### 例题

> [矩阵快速冪]()
>
> [矩阵加速数列]()
>
> [随机数生成器]()
>
> [1962 斐波那契数数列]()

```cpp
void mul ()
{
    int fa[105][105]={0};
    rep(i,n) rep(j,n) rep(k,n)
    fa[i][j]=(fa[i][j]+syc[i][k]*syc[k][j])%inf;
    rep(i,n) rep(j,n) syc[i][j]=fa[i][j]; 
}
void juzhen(ll a)
{
    rep(i,n) ans[i][i]=1;
    ll fa[105][105];
    while(a)
    {
      if(a%2==1) 
      {
        rep(i,n) rep(j,n) fa[i][j]=ans[i][j],ans[i][j]=0;
        rep(i,n) rep(j,n) rep(k,n)
        ans[i][j]=(ans[i][j]+fa[i][k]*syc[k][j])%inf;
     }
      rep(i,n) rep(j,n) fa[i][j]=syc[i][j],syc[i][j]=0;
      rep(i,n) rep(j,n) rep(k,n)
      syc[i][j]=(syc[i][j]+fa[i][k]*fa[k][j])%inf;
      a>>=1;
    }
}
```
## 排列组合
### 杨辉三角（组合数）

#### 组合数公式

#### 递推公式

### 二项式定理



# 模拟
## 高精度



# STL
## 容器
### vector<>
### map<>
### set<>
## 算法
## sort
```cpp
#include <algorithm>
using namespace std;//如果想用sort那么一定加上以上这两行
#define maxn 10000
int a[maxn+5];//定义a数组，大小为maxn+5
struct jgt
{
    int x,y;
}px[maxn+5];//定义结构体
bool cmp(jgt p,jgt q)//定义要排序的类型如某结构体
{
    return p.x>q.x;//以x排序
}
bool cmp(jgt p,jgt q)
{
    if(p.x>q.x)  return 1;
    else if(p.x<q.x)  return 0;
    else
    {
        if(p.y>q.y)  return 1;
        else  return 0;
    }//以x为第一优先，以y为第二优先排序
}
bool cmp(jgt p,jgt q)
{
    return p.x==q.x?p.y>q.y:p.x>q.x;
}//此种写法为三目运算符，与上一种等价
.
.
.

sort(a+1,a+maxn+1);//sort中三个参量，第一个为数组排序的起始地址，
                   //第二个是结束的地址（最后一位要排序的地址的下一地址），
                   //第三个参数是排序的方法，可以是从大到小也可是从小到大，还可以不写第三个参数，此时默认的排序方法是从小到大排序。
sort(px+1,px+maxn+1,cmp);//cmp即为排序方式
```

# 优化
## 离散化
```cpp
int itv[maxn], cnt;
int inline vti(int v) { return lower_bound(itv, itv + cnt, v) - itv; }
```