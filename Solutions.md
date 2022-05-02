# 剑指 offer

## T3 数组中重复出现的数

使用哈希表记录到目前为止已经出现过的数，如果当前数能被查询到，说明在前面出现过，返回该数，否则把这个数也记录为出现过。

```java
class Solution {
    public int findRepeatNumber(int[] nums) {
        Map<Integer, Boolean> appear = new HashMap<>();
        for(int x : nums){
            if(appear.get(x)!= null && appear.get(x))return x;
            appear.put(x, true);
        }

        return -1;
    }
}
```

复杂度分析 : 

```
Time : O(n) // scan array nums
Space : O(n) // hashmap
```

## T4 二维数组中的查找

法一 : 在每一行里面进行二分查找

```
Time : O(mlogn)
Space : O(1)
```



法二 : 

根据所给定的数据结构组织查找的顺序，类似二分的思路，每次比较大小后要排除一些区域。

从右上角开始查找，如果 `target == matrix[x][y]`, 那么查找到目标，如果 `target > matrix[x][y]`, 那么 `matrix[x][y]` 所在行其他元素也被排除(因为 `matrix[x][y]` 是这行里面最大的，仍然小于target), 同理, 如果 `matrix[x][y] > target`, 其所在列都被排除。如果所有区域都被排除仍然没找到，说明找不到目标值

```java
class Solution {
    public boolean findNumberIn2DArray(int[][] matrix, int target) {
        int rows = matrix.length;
        if(rows == 0)return false;
        int cols = matrix[0].length;
        if(cols == 0)return false;

        int x = 0, y = cols - 1;
        while(x < rows && y >= 0){
            if(matrix[x][y] == target)return true;
            if(matrix[x][y] > target)y --;
            else x ++;
        }

        return false;
    }
}
```

复杂度分析:
```
Time : O(m+n) //最多排除这么多行和列
Space : O(1)
```

## T5 替换空格

使用 StringBuilder 来拼接，防止创建多个字符串，碰到非空格直接拼，碰到空格就换成 %20

```java
class Solution {
    public String replaceSpace(String s) {
        StringBuilder str = new StringBuilder();
        String replaceStr = "%20";
        for(int i = 0; i < s.length(); i ++){
            if(s.charAt(i) != ' ')str.append(s.charAt(i));
            else str.append(replaceStr);
        }

        return str.toString();
    }
}
```

## T6 从尾到头打印链表

从头到尾遍历链表，遍历到的元素依次压栈，遍历结束后依次弹出放到答案数组中。

```java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode(int x) { val = x; }
 * }
 */
class Solution {
    public int[] reversePrint(ListNode head) {
        Deque<Integer> stk = new LinkedList();
        while(head != null){
            stk.push(head.val);
            head = head.next;
        }
        int[] res = new int[stk.size()];
        int count = 0;
        while(!stk.isEmpty()){
            res[count++] = stk.peek();
            stk.pop();
        }
        return res;
    }
}
```

复杂度分析:
```
Time : O(n) // scan array
Space : O(n) // stack
```

## T30 包含min函数的栈

首先需要一个栈，用数组模拟一个即可，然后用一个 minv 数组记录栈底到每个位置最小的数。

```java
class MinStack {
    private int[] stk;
    private int[] minv;
    private int esp;
    /** initialize your data structure here. */
    public MinStack() {
        stk = new int[20010];
        minv = new int[20010];
        esp = 0;
    }
    
    public void push(int x) {
        stk[++esp] = x;
        if(esp == 1)minv[esp] = x;
        else minv[esp] = Math.min(minv[esp-1], x);
    }
    
    public void pop() {
        --esp;
    }
    
    public int top() {
        return stk[esp];   
    }
    
    public int min() {
        return minv[esp];
    }
}

/**
 * Your MinStack object will be instantiated and called as such:
 * MinStack obj = new MinStack();
 * obj.push(x);
 * obj.pop();
 * int param_3 = obj.top();
 * int param_4 = obj.min();
 */
```

复杂度分析：

```
Time:
push : O(1)
pop : O(1)
top() : O(1)
min() : O(1)
Space : O(n)
```

