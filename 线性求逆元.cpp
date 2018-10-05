//syc 线性求逆元  高效求单个数逆元使用Ex_gcd
//P3811
#include<iostream>
#include<cstdio>
using namespace std;

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
