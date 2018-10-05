// syc 已经死了的spfa
// 洛谷 P3371
#include<iostream>
#include<cstdio>
#include<algorithm>
#include<queue>
#define inf 2147483647
using namespace std;

const int maxn = ;
const int maxm = ;

struct edge{
	int from,to,val,nxt;
}e[maxm];

int d[maxn],head[maxn],vis[maxn];

int n,m,cnt,start,end;

queue <int> Q;

void add(int x,int y,int z){
	e[++cnt].from = x;
	e[cnt].to = y;
	e[cnt].val = z;
	e[cnt].nxt = head[x];
	head[x] = cnt;
}

void in(){
	scanf("%d%d%d",&n,&m,&start);
	int a,b,c;
	for(int i = 1 ; i <= m ; i++){
		scanf("%d%d%d",&a,&b,&c);
		add(a,b,c);
	}
}

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

void out(){
	for(int i = 1 ; i <= n ; i++) printf("%d ",d[i]);
}

int main(){
	in();
	spfa();
	out();
	return 0;
}
