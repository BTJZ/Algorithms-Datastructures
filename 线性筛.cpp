//syc 欧拉筛法
//P3383 
#include<cstdio>
#include<iostream>
#include<cstring>
using namespace std;

const int maxn = 10000000 + 5;

int prime[maxn];

int flag[maxn];

int cnt,n,m;

int main(){
    cin>>n>>m;
    flag[1] = 1;
    for(int i = 2 ; i <= n ; i++){
        if(!flag[i]) prime[++cnt] = i;
        for(int j = 1 ; j <= cnt && prime[j]*i <= n ; j++){
            flag[i*prime[j]] = 1;
            if(i%prime[j]==0) break;
        }
    }
    int qus;
    for(int i = 1 ; i <= m ; i++){
        scanf("%d",&qus);
        if(flag[qus]) printf("No\n");
        else printf("Yes\n");
    }
    return 0;
}
