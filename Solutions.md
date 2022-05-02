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

## T24 反转链表

递归做法，先反转以第二个节点开头的链表，然后把第二个节点的next指向第一个节点，第一个节点next置为空，返回新的头节点。

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
    public ListNode reverseList(ListNode head) {
        if(head == null || head.next == null)return head;
        ListNode h = reverseList(head.next);
        head.next.next = head;
        head.next = null;
        return h;
    }
}
```

复杂度分析 :

```
Time : O(n) 
Space : O(n) // recursion stack
```

## T35 复杂链表的复制

扫描两趟原链表，第一趟复制节点和 next 关系，并记录原链表到新链表节点之间的一一映射关系（用 hashmap），第二趟利用映射关系复制random 关系。

```java
/*
// Definition for a Node.
class Node {
    int val;
    Node next;
    Node random;

    public Node(int val) {
        this.val = val;
        this.next = null;
        this.random = null;
    }
}
*/
class Solution {
    public Node copyRandomList(Node head) {
        Node dummy = new Node(-1);
        Node cur = dummy;
        Node backUp = head;
        Map<Node, Node> mapping = new HashMap<>();
        while(head != null){
            Node n = new Node(head.val);
            mapping.put(head, n);
            cur.next = n;
            cur = cur.next;
            head = head.next;
        }
        cur = dummy.next;
        while(backUp != null){
            if(backUp.random != null)cur.random = mapping.get(backUp.random);
            backUp = backUp.next;
            cur = cur.next;
        }

        return dummy.next;
    }
}
```

复杂度分析：

```
Time:O(n) // scan twice
Space: O(n) // hashmap
```

## T58 剑指offer 左旋转字符串

简单做法

```java
class Solution {
    public String reverseLeftWords(String s, int n) {
        return s.substring(n) + s.substring(0, n);
    }
}
```

如果不能用 `substring`, 要用 `StringBuilder` 对象来拼接字符串，而不是直接拼接。(字符串常量拼接会创建多个对象，效率低)

```java
class Solution {
    public String reverseLeftWords(String s, int n) {
        StringBuilder str = new StringBuilder();
        for(int i = n; i < s.length(); i ++)str.append(s.charAt(i));
        for(int i = 0; i < n; i ++)str.append(s.charAt(i));

        return str.toString();
    }
}
```

## T53 在排序数组中查找数字

二分查找，找到第一个出现的地方和最后一个出现的地方，出现次数 就是 last - first + 1

```java
class Solution {
    public int search(int[] nums, int target) {
        int len = nums.length;
        if(len == 0)return 0;
        int first = findFirst(nums, target);
        if(nums[first] != target)return 0;
        
        int last = findLast(nums, target);

        return last - first + 1;
    }

    private int findFirst(int[] nums, int x){
        int l = 0, r = nums.length-1;
        while(l < r){
            int mid = l+r>>1;
            if(nums[mid] >= x)r = mid;
            else l = mid + 1;
        }

        return l;
    }
    private int findLast(int[] nums, int x){
        int l = 0, r = nums.length-1;
        while(l < r){
            int mid = l+r+1>>1;
            if(nums[mid] <= x)l = mid;
            else r = mid - 1;
        }

        return l;
    }
}
```

二分，对数时间复杂度

## T53 0-n-1 中缺失的数字

观察区间的二分性质，前一半区间满足 i == nums[i] , 后一半区间满足 nums[i] > i, 后一半区间可能为空，此时缺失的就是最后一个位置对应的数。用这个性质，查找第一个满足 nums[i] > i 的下标就是答案，如果查找不到，那么说明是后一半区间为空的情况。

```java
class Solution {
    public int missingNumber(int[] nums) {
        int l = 0, r = nums.length-1;
        while(l < r){
            int mid = l+r>>1;
            if(nums[mid] > mid)r = mid;
            else l = mid + 1;
        }

        return nums[l] == l ? l+1 : l;
    }
}
```

二分，对数时间复杂度

