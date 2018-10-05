//syc LCA RMQ 
//洛谷 P1967 
#include<iostream>
#include<cstdio>
#include<algorithm>
#define inf 0x3f3f3f3f
using namespace std;

const int maxn = 10000 + 5;
const int maxm = 50000 + 5;

int fa[maxn];
int n,m,q;
int head[maxn],head2[maxn],cnt,cnt2,num;
int vis[maxn],lca1[20][maxn],lca2[20][maxn],deep[maxn];
int bit[20],lg[maxn];

//lca1数组 记录father
//lca2数组 记录区间最小值

struct edge{
    int from,to,val,nxt;
}e[maxm*2];

struct edge2{
    int from,to,nxt,val;
}e2[maxn*2];

bool cmp(edge x, edge y){
    return x.val>y.val;
}

void add1(int x,int y,int z){
    e[++cnt].from = x;
    e[cnt].to = y;
    e[cnt].val = z;
    e[cnt].nxt = head[x];
    head[x] = cnt;
}

void add2(int x,int y,int z){
    e2[++cnt2].from = x;
    e2[cnt2].to = y;
    e2[cnt2].val = z;
    e2[cnt2].nxt = head2[x];
    head2[x] = cnt2;
}

int find(int x){
    if(x==fa[x]) return x;
    return fa[x] = find(fa[x]);
}

void Union(int x,int y){
    fa[x] = y;
}

void ku(){
    int node = 0,k = 0;
    for(int i = 1 ; i <= cnt ; i++){
        int x,y;
        x = find(e[++node].from);
        y = find(e[node].to);
        if(x==y) continue;
        else{
            Union(x,y);
            add2(e[node].from,e[node].to,e[node].val);
            add2(e[node].to,e[node].from,e[node].val);
            k++;
            if(k==n-1) break;
        }
    }
}

void dfs(int f,int father){
    if(vis[f]==1&&father==0) return;
    vis[f] = 1;
    deep[f] = deep[father] + 1;
    lca1[0][f] = father;
    if(father!=0){
        for(int i = head2[f] ; i ; i = e2[i].nxt) if(e2[i].to==father){
            lca2[0][f] = e2[i].val;
            break;
        } 
    }
    for(int i = 1 ; bit[i] <= deep[f] ; i++) lca1[i][f] = lca1[i-1][ lca1[i-1][f] ] ;
    for(int i = 1 ; bit[i] <= deep[f] ; i++){
        lca2[i][f] = min(lca2[i-1][f],lca2[i-1][ lca1[i-1][f] ]);
    }
    for(int i = head2[f] ; i ; i = e2[i].nxt){
        if(e2[i].to!=father) dfs(e2[i].to,f);
    } 
}

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

int main(){
    scanf("%d%d",&n,&m);
    bit[0] = 1;
    for(int i = 1 ; i <= 17 ; i++)  bit[i] = bit[i-1] * 2;
    for(int i=1;i<=n;i++)  lg[i]=lg[i-1]+(1<<lg[i-1]==i);
    int a,b,c;
    for(int i = 1 ; i <= m ; i++){
        scanf("%d%d%d",&a,&b,&c);
        add1(a,b,c);
    }
    for(int i = 1 ; i <= n ; i++)fa[i] = i;
    sort(e+1,e+cnt+1,cmp);
    ku();
    for(int i = 1 ; i <= n ; i++) 
    dfs(i,0);
    scanf("%d",&q);
    for(int i = 1 ; i <= q; i++){
        scanf("%d%d",&a,&b);
        if(find(a)!=find(b))
        cout<<-1<<endl;
        else
        cout<<lca(a,b)<<endl;
    }
    return 0;
}
