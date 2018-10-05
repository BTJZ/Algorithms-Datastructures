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



//排序例题详见洛谷排序试炼场

