int itv[maxn], cnt;
int inline vti(int v) { return lower_bound(itv, itv + cnt, v) - itv; }
