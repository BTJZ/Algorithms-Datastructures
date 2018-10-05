// 矩阵快速幂 
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
// 带权并查集
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
// tarjin
void tarjin(int t)
 {
    syc++;
    dfn[t]=syc,low[t]=syc;
    stack[++size]=t,instack[t]=1;
    for(int i=head[t];i;i=way[i].fa)
     {
        if(!dfn[way[i].ed]) 
        {
           tarjin(way[i].ed);
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
//前缀和与差分 
void a()
 {
   int a[maxn],n,b[maxn];
   n=re();
   f(i,1,n) a[i]=re();
   f(i,1,n) b[i]+=b[i-1];
   f(i,1,n) b[i]-=b[i-1];
 }
