#include <bits/stdc++.h>
using namespace std;

class Solution
{
public:
    /**
     * Note: 类名、方法名、参数名已经指定，请勿修改
     *
     *
     *
     * @param w int整型  卡车的载重量
     * @param c int整型 vector 每个集装箱的重量
     * @return int整型vector
     */
    vector<int> trunkLoad(int w, vector<int> &c)
    {
        // write code here
        map<int, int> map1, map2;
        vector<int> ans;
        int len = 0;
        int youxiao[100010];
        for (int i = 0; i < c.size(); i++)
        {
            int len2 = len;
            for (int j = 1; j <= len2; j++)
            {
                int mb = youxiao[j] + c[i];
                if (map1[mb] == 0)
                {
                    map1[mb] = map1[youxiao[j]] + 1;
                    map2[mb] = c[i];
                    len++;
                    youxiao[len] = mb;
                }
                else
                {
                    if (map1[mb] > map1[youxiao[j]] + 1)
                    {
                        map1[mb] = map1[youxiao[j]] + 1;
                        map2[mb] = c[i];
                    }
                }
            }
            if (map1[c[i]] == 0)
            {
                map1[c[i]] = 1;
                map2[c[i]] = c[i];
                len++;
                youxiao[len] = c[i];
            }
            // cout << c[i] << "!!!!!" << endl;
            // for (int j = 1; j <= len; j++)
            // {
            //     cout << youxiao[j] << " " << map1[youxiao[j]] << " " << map2[youxiao[j]] << endl;
            // }
        }
        map<int, int> a;
        int pre = w;
        while (pre != 0)
        {
            // cout << pre << " " << map1[pre] << " " << map2[pre] << endl;
            a[map2[pre]] = 1;
            pre = pre - map2[pre];
        }
        for (int i = 0; i < c.size(); i++)
            if (a[c[i]] == 1)
                ans.push_back(1);
            else
                ans.push_back(0);
        return ans;
    }
};

int main()
{
    Solution solution;
    int w = 10;
    vector<int> c = {5, 2, 6, 4, 3};
    vector<int> result = solution.trunkLoad(w, c);
    for (int i : result)
    {
        cout << i << " ";
    }
    cout << endl;
    return 0;
}