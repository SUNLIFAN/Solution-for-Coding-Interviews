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

## T50 第一个只出现一次的字符

两次遍历，第一次计数，第二次查看。

```java
class Solution {
    public char firstUniqChar(String s) {
        Map<Character, Integer> count = new HashMap<>();
        for(int i = 0; i < s.length(); i ++){
            if(count.get(s.charAt(i)) == null)count.put(s.charAt(i), 1);
            else {
                int cnt = count.get(s.charAt(i)) + 1;
                count.put(s.charAt(i), cnt);
            }
        }

        for(int i = 0; i < s.length(); i ++){
            if(count.get(s.charAt(i)) == 1)return s.charAt(i);
        }

        return ' ';
    }
}
```

## T11 旋转数组的最小数字

观察区间的二分性质，如果没有重复元素的话，那么前一半区间都是大于等于 number[0] 的元素，后一半区间都是小于 numbers[0] 的元素，用二分查找第一个小于 numbers[0] 的元素即可。对于有重复元素的情况，可能出现这样的情况，尾部的一些元素和numbers[0] 相等，只要先把这些元素去掉就可以按照无重复元素的做法来做。（因为依然满足上述的区间性质）

```java
class Solution {
    public int minArray(int[] numbers) {
        int len = numbers.length;
        while(len > 1 && numbers[len-1] == numbers[0])len --;
        int l = 0, r = len-1;
        while(l < r){
            int mid = l+r>>1;
            if(numbers[mid] < numbers[0])r = mid;
            else l = mid + 1;
        }

        return numbers[l] < numbers[0] ? numbers[l] : numbers[0];
    }
}
```

## T32 从上到下打印二叉树

简单 BFS

```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */
class Solution {
    public int[] levelOrder(TreeNode root) {
        List<Integer> res = new ArrayList<>();
        if(root == null)return new int[0];
        Deque<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        while(!queue.isEmpty()){
            TreeNode frt = queue.poll();
            res.add(frt.val);
            if(frt.left != null)queue.offer(frt.left);
            if(frt.right != null)queue.offer(frt.right);
        }

        int[] ans = new int[res.size()];
        for(int i = 0; i < res.size(); i ++){
            ans[i] = res.get(i);
        }

        return ans;
    }
}
```

## T26 树的子结构

判断是否是树的子结构有点像判断是否是字符串子串，考虑两种情况，一种是 B 是 A 的左子树或者右子树的子结构，一种是从根开始匹配的子结构。第一种情况可以递归解决，第二种情况写个 dfs 即可。

```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */
class Solution {
    public boolean isSubStructure(TreeNode A, TreeNode B) {
        if(A == null)return false;
        if(B == null)return false;

        if(dfs(B, A))return true;

        return isSubStructure(A.left, B) || isSubStructure(A.right, B);
    }

    /* a is sub of b ? 
     */
    boolean dfs(TreeNode a, TreeNode b){
        if(b == null)return false;
        if(a.left == null && a.right == null){
            return a.val == b.val;
        }
        if(a.val != b.val)return false;
        boolean res = true;
        if(a.left != null)res =  res && dfs(a.left, b.left);
        if(a.right != null)res = res && dfs(a.right, b.right);

        return res; 
    }
}
```

复杂度分析:

```
类比字符串匹配，这是暴力写法, 其中 n 是节点个数
Time : O(n^2) 
Space : O(n) \\ recursion
```

## T27 二叉树的镜像

递归，分两步，先把左右子树做镜像，然后交换左右子树。

```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */
class Solution {
    public TreeNode mirrorTree(TreeNode root) {
        if(root == null || root.left ==null && root.right==null)return root;

        TreeNode tmp = mirrorTree(root.left);
        root.left = mirrorTree(root.right);
        root.right = tmp;

        return root;
    }
}
```

复杂度分析：

```
Time : O(n) \\ scan every node
Space : O(n) \\ recursion
```

## T28 对称的二叉树

法一：遍历

获得左子树的前序遍历序列和右子树的对称前序遍历序列，如果相等的话那么左右子树对称。

```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */
class Solution {
    public boolean isSymmetric(TreeNode root) {
        if(root == null)return true;
        List<Integer> pre = new ArrayList<>();
        List<Integer> symPre = new ArrayList<>();
        preorder(root.left, pre);
        symPreorder(root.right, symPre);
        for(int i = 0; i < pre.size(); i ++){
            if(pre.get(i) != symPre.get(i))return false;
        }

        return true;
    }

    void preorder(TreeNode root, List<Integer> res){
        if(root == null){
            res.add(null);
            return;
        }
        res.add(root.val);
        preorder(root.left, res);
        preorder(root.right, res);
    }

    void symPreorder(TreeNode root, List<Integer> res){
        if(root == null){
            res.add(null);
            return;
        }

        res.add(root.val);
        symPreorder(root.right, res);
        symPreorder(root.left, res);
    }
}
```

法二： 递归

```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */
class Solution {
    public boolean isSymmetric(TreeNode root) {
        if(root == null)return true;

        return check(root.left, root.right);
    }

    public boolean check(TreeNode p, TreeNode q){
        if(p == null && q == null)return true;
        if(p == null || q == null)return false;

        return p.val == q.val && check(p.left, q.right) && check(p.right, q.left);
    }
}

```

## T12 矩阵中的路径

回溯

```java
class Solution {
    private int[] dx = {1, -1, 0, 0};
    private int[] dy = {0, 0, 1, -1};
    private boolean[][] visited;

    public boolean exist(char[][] board, String word) {
        visited = new boolean[board.length][board[0].length];
        for(int i = 0; i < visited.length; i ++)
            for(int j = 0; j < visited[0].length; j ++)
                visited[i][j] = false;
        for(int i = 0; i < board.length; i ++)
            for(int j = 0; j < board[0].length; j ++){
                if(board[i][j] == word.charAt(0)){
                    visited[i][j] = true;
                    boolean res = dfs(board, word, 1, i, j);
                    visited[i][j] = false;
                    if(res)return true;
                }
            }
        return false;
    }

    public boolean dfs(char[][] board, String word, int hasMatched, int x, int y){
        if(hasMatched == word.length())return true;
        for(int i = 0; i < 4; i ++){
            int x_ = x + dx[i], y_ = y + dy[i];
            if(x_>=0 && x_ < board.length && y_>=0 && y_ <board[0].length && !visited[x_][y_]){
                if(board[x_][y_] == word.charAt(hasMatched)){
                    visited[x_][y_] = true;
                    boolean res = dfs(board, word, hasMatched+1, x_, y_);
                    visited[x_][y_] = false;
                    if(res)return true;
                }
            }
        }

        return false;
    }
}

```

