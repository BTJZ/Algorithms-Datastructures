//syc LCA
//洛谷 P3379
#include<iostream>
#include<cstdio>
#include<algorithm>
#include<cmath>
using namespace std;

const int maxn = 500000 + 5;

struct edge{
    int from,to,next;
}edge[2*maxn];

int lg[maxn],deep[maxn],lca[20][maxn],head[maxn],bit[25];

int n,m,s,num;

void add(int x , int y){
    edge[++num].from = x;
    edge[num].to = y;
    edge[num].next = head[x];
    head[x] = num;
}

void in(){
	scanf("%d%d%d",&n,&m,&s);
    bit[0] = 1;
    for(int i = 1 ; i <= 20 ; i++) bit[i] = bit[i-1] * 2;
    for(int i = 2;i <= n;i++)  lg[i] = lg[i>>1] + 1;
    for(int i = 1 ; i <= n-1 ; i++){
        int a,b;
        scanf("%d%d",&a,&b);
        add(a,b);
        add(b,a);
    }
}

void dfs(int f,int father){
	lca[0][f] = father;
	deep[f] = deep[father] + 1;
	int d = deep[f];
	for(int i = 1 ; i <= lg[d] ; i++) lca[i][f] = lca[i-1][ lca[i-1][f] ];
	for(int i = head[f] ; i ; i = edge[i].next) if(edge[i].to!=father) dfs(edge[i].to,f);
}

int Lca(int x,int y){
	if(deep[x]<deep[y]) swap(x,y);
	while(deep[x]!=deep[y]){
		int d = deep[x] - deep[y];
		x = lca[lg[d]][x];
	}
	if(x==y) return x;
	for(int i = lg[deep[x]] ; i>=0 ; i--){
		if(lca[i][x]!=lca[i][y]) x = lca[i][x] , y = lca[i][y];
	}
	return lca[0][x];
}

void out(){
	int a,b;
	for(int i = 1 ; i <= m ; i++){
		scanf("%d%d",&a,&b);
		printf("%d\n",Lca(a,b));
	}
}

int main(){
	in();
	dfs(s,0);
	out();
	return 0;
}
