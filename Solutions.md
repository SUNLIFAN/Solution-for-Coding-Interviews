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

## T13 机器人的运动范围

可达性的问题，BFS 即可

```java
class Solution {
    private int[] dx = {1, -1, 0, 0};
    private int[] dy = {0, 0, 1, -1};
    private boolean[][] visited;
    public int movingCount(int m, int n, int k) {
        visited = new boolean[m][n];
        for(int i = 0; i < m; i ++)
            for(int j = 0; j < n; j ++)
             visited[i][j] = false;
        int count = 1;
        Deque<Point> q = new LinkedList<>();
        q.offer(new Point(0, 0));
        visited[0][0] = true;
        while(!q.isEmpty()){
            Point p = q.poll();
            int x = p.x, y = p.y;
            for(int i = 0; i < 4; i ++){
                int x_ = x + dx[i], y_ = y + dy[i];
                if(x_ >= 0 && x_ < m && y_ >= 0 && y_ < n && !visited[x_][y_]){
                    if(getDigitSum(x_) + getDigitSum(y_) <= k){
                        q.push(new Point(x_, y_));
                        visited[x_][y_] = true;
                        count ++;
                    }
                }
            }
        }

        return count;        
    }

    private int getDigitSum(int num){
        int sum = 0;
        while(num > 0){
            sum = sum + num % 10;
            num /= 10;
        }

        return sum;
    }

    private class Point {
        int x;
        int y;
        Point(int x, int y){
            this.x = x;
            this.y = y;
        }
    }
}

```

## T34 二叉树中和为某一值的路径

回溯

```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode() {}
 *     TreeNode(int val) { this.val = val; }
 *     TreeNode(int val, TreeNode left, TreeNode right) {
 *         this.val = val;
 *         this.left = left;
 *         this.right = right;
 *     }
 * }
 */
class Solution {
    List<List<Integer>> res = new ArrayList<>();
    public List<List<Integer>> pathSum(TreeNode root, int target) {
        dfs(root, 0, target, new LinkedList<>());

        return res;
    }

    private void dfs(TreeNode root, int curSum, int target, LinkedList<Integer> tmp){
        if(root == null)return;
        if(root.left == null && root.right == null){
            curSum += root.val;
            if(curSum == target){
                tmp.add(root.val);
                res.add(new ArrayList(tmp));
                tmp.removeLast();
                return;
            }
        }

        tmp.add(root.val);
        curSum += root.val;
        if(root.left != null)dfs(root.left, curSum, target, tmp);
        if(root.right != null)dfs(root.right, curSum, target, tmp);
        tmp.removeLast();
    }

}

```

```java
lst2 = new ArrayList(lst1); 是浅拷贝
```

## T54 二叉搜索树的第 k 大节点

二叉搜索树的中序遍历是递增序列，求出之和倒数第 k 个元素就是第 k 大

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
    public int kthLargest(TreeNode root, int k) {
        List<Integer> lst = new ArrayList<>();
        dfs(lst, root);
        return lst.get(lst.size()-k);
    }

    void dfs(List<Integer> lst, TreeNode root){
        if(root == null)return;
        if(root.left == null && root.right == null){
            lst.add(root.val);
            return;
        }
        dfs(lst, root.left);
        lst.add(root.val);
        dfs(lst, root.right);
    }
}
```

## T55 二叉树的深度

简单递归

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
    public int maxDepth(TreeNode root) {
        if(root == null)return 0;
        if(root.left == null && root.right == null)return 1;

        return Math.max(maxDepth(root.left), maxDepth(root.right)) + 1;
    }
}
```

每个节点都会遍历到，时间复杂度是 `O(n)`,最坏情况是极度不平衡的链形的二叉树，递归栈深度 `O(n)`

## T15 二进制中 1 的个数

每次获取二进制中最低位 1, 并减去，直到为 0, 同时计数。

```java
public class Solution {
    // you need to treat n as an unsigned value
    public int hammingWeight(int n) {
        int cnt = 0;
        while(n != 0){
            n -= lowbit(n);
            cnt ++;
        }

        return cnt;
    }

    private int lowbit(int n){
        return n & (-n);
    }
}
```

## T64 求 1 + 2 + ... + n

对使用的运算符做了一些限制，感觉是脑筋急转弯，用与运算的短路性质实现了 if, else 的功能。Java 对表达式的书写要求比较严格, 整型不能自动转成布尔类型，比 C++麻烦一点。

```java
class Solution {
    public int sumNums(int n) {
        int res = n;
        boolean flag = (n > 0) && (res += sumNums(n-1)) > 0;
        return res;
    }
}

```

## T55 平衡二叉树

一棵树是平衡二叉树，当它的左右子树高度差不超过 1, 且左右子树也是平衡二叉树，利用定义递归做即可。

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
    Map<TreeNode, Integer> map = new HashMap<>();
    public boolean isBalanced(TreeNode root) {
        if(root == null)return true;
        dfs(root);

        return check(root);
    }

    private boolean check(TreeNode root){
        if(root == null)return true;
        
        int lHeight = root.left == null ? 0 : map.get(root.left);
        int rHeight = root.right == null ? 0 : map.get(root.right);

        return Math.abs(lHeight - rHeight) <= 1 && 
        check(root.left) && check(root.right);
    }

    private void dfs(TreeNode root){
        if(root == null)return;
        if(root.left == null && root.right == null){
            map.put(root, 1);
            return;
        }
        dfs(root.left);
        dfs(root.right);
        int lHeight = root.left == null ? 0 : map.get(root.left);
        int rHeight = root.right == null ? 0 : map.get(root.right);
        int height = Math.max(lHeight, rHeight) + 1;
        map.put(root, height);

        return;
    }
}

```

复杂度分析: 

```
Time : O(n) //每个节点遍历到常数次
Space : O(n) // Map, recursion
```

## T32 从上到下打印二叉树 II

广搜，先存遍历序列和对应深度，再获取最终答案列表。

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
    public List<List<Integer>> levelOrder(TreeNode root) {
        List<List<Integer>> res = new ArrayList<>();
        List<Node> lst = new ArrayList<>();
        if(root == null)return res;
        Deque<Node> q = new LinkedList<>();
        Node r = new Node(root, 0);
        q.offer(r);
        lst.add(r);
        while(!q.isEmpty()){
            Node frt = q.poll();
            if(frt.node.left != null){
                Node ln = new Node(frt.node.left, frt.depth + 1);
                lst.add(ln);
                q.offer(ln);
            }
            if(frt.node.right != null){
                Node rn = new Node(frt.node.right, frt.depth + 1);
                lst.add(rn);
                q.offer(rn);
            }
        }

        int prevDepth = -1;
        List<Integer> tmp = new ArrayList<>();
        for(int i = 0; i < lst.size(); i ++){
            if(lst.get(i).depth > prevDepth){
                prevDepth = lst.get(i).depth;
                if(prevDepth > 0)res.add(new ArrayList(tmp));
                tmp.clear();
            }
            tmp.add(lst.get(i).node.val);
        }
        res.add(tmp);

        return res;
    }

    private class Node {
        TreeNode node;
        int depth;
        Node(TreeNode n, int d){
            node = n;
            depth = d;
        }
    }
}

```

## T32 从上到下打印二叉树

和上一题基本一样，用一个变量记录当前行是否需要 reverse 即可。

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
    
        public List<List<Integer>> levelOrder(TreeNode root) {
        List<List<Integer>> res = new ArrayList<>();
        List<Node> lst = new ArrayList<>();
        if(root == null)return res;
        Deque<Node> q = new LinkedList<>();
        Node r = new Node(root, 0);
        q.offer(r);
        lst.add(r);
        while(!q.isEmpty()){
            Node frt = q.poll();
            if(frt.node.left != null){
                Node ln = new Node(frt.node.left, frt.depth + 1);
                lst.add(ln);
                q.offer(ln);
            }
            if(frt.node.right != null){
                Node rn = new Node(frt.node.right, frt.depth + 1);
                lst.add(rn);
                q.offer(rn);
            }
        }

        int prevDepth = -1;
        boolean needReverse = false;
        List<Integer> tmp = new ArrayList<>();
        for(int i = 0; i < lst.size(); i ++){
            if(lst.get(i).depth > prevDepth){
                prevDepth = lst.get(i).depth;
                if(prevDepth > 0){
                    if(needReverse)Collections.reverse(tmp);
                    needReverse = !needReverse;
                    res.add(new ArrayList(tmp));
                }
                tmp.clear();
            }
            tmp.add(lst.get(i).node.val);
        }
        if(needReverse)Collections.reverse(tmp);
        res.add(tmp);

        return res;
    }

    private class Node {
        TreeNode node;
        int depth;
        Node(TreeNode n, int d){
            node = n;
            depth = d;
        }
    }
}

```

## T36 二叉搜索树和双向链表

先对左右子树递归，然后再拼起来

```java
/*
// Definition for a Node.
class Node {
    public int val;
    public Node left;
    public Node right;

    public Node() {}

    public Node(int _val) {
        val = _val;
    }

    public Node(int _val,Node _left,Node _right) {
        val = _val;
        left = _left;
        right = _right;
    }
};
*/
class Solution {
    public Node treeToDoublyList(Node root) {
        if(root == null)return root;
        if(root.left == null && root.right == null){
            root.left = root.right = root;
            return root;
        }
        Node leftHead = treeToDoublyList(root.left);
        Node rightHead = treeToDoublyList(root.right);
        Node head = leftHead == null ? root : leftHead;
        Node tail = rightHead == null ? root : rightHead.left;
        if(leftHead != null){
            Node cur = leftHead;
            while(cur.right != leftHead){
                cur = cur.right;
            }
            cur.right = root;
            root.left = cur;
        }
        if(rightHead != null){
            rightHead.left = root;
            root.right = rightHead;
        }
        
        head.left = tail;
        tail.right = head;

        return head;
    }
}

```

## T21 调整数组顺序使得奇数位于偶数前面

仿照快速排序的 partition 算法, 把奇数留在左边，偶数都扔到后面。

```java
class Solution {
    public int[] exchange(int[] nums) {
        int i = -1, j = nums.length;
        while(i < j){
            do i++;while(i < nums.length && (nums[i] & 1) == 1);
            do j--;while(j >= 0 && (nums[j] & 1) == 0);
            if(i < j){
                swap(nums, i, j);
            }
        }

        return nums;
    }

    private void swap(int[] nums, int i, int j){
        int tmp = nums[i];
        nums[i] = nums[j];
        nums[j] = tmp;
    }
}
```

## T18 删除链表的节点

为了减少判断，先新建一个 dummy node, 然后从 dummy node 开始查找待删除链表节点的前一个节点，找到之后将它的 next 置为待删除节点的next

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
    public ListNode deleteNode(ListNode head, int val) {
        ListNode dummy = new ListNode(-1);
        dummy.next = head;
        ListNode cur = dummy;
        while(cur.next != null && cur.next.val != val){
            cur = cur.next;
        }
        if(cur.next != null)cur.next = cur.next.next;

        return dummy.next;
    }
}

```

## T22 链表中倒数第 k 个节点

法一: 遍历链表，节点逐个压栈, 弹出 k-1 个，此时栈顶就是倒数第 k 个节点。

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
    public ListNode getKthFromEnd(ListNode head, int k) {
        Deque<ListNode> stk = new LinkedList<>();
        while(head != null){
            stk.push(head);
            head = head.next;
        }

        for(int i = 0; i < k-1; i ++)stk.pop();

        return stk.peek();
    }
}

```

法二：双指针

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
    public ListNode getKthFromEnd(ListNode head, int k) {
        ListNode front = head;
        while(front != null && k > 0){
            front = front.next;
            k --;
        }
        if(k > 0)return null;
        ListNode back = head;
        while(front != null){
            front = front.next;
            back = back.next;
        }

        return back;
    }
}

```

两个做法时间都是线性，法一空间也是线性。

## T25 合并两个排序链表

双指针归并

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
    public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
        if(l1 == null && l2 == null)return null;
        if(l1 == null || l2 == null)return l1 == null ? l2 : l1;

        ListNode dummy = new ListNode(-1);
        ListNode cur = dummy;
        while(l1!= null && l2 != null){
            if(l1.val <= l2.val){
                cur.next = l1;
                l1 = l1.next;
            }else {
                cur.next = l2;
                l2 = l2.next;
            }
            cur = cur.next;
        }
        if(l1 != null)cur.next = l1;
        if(l2 != null)cur.next = l2;

        return dummy.next;
    }
}

```

时间复杂度 `O(n)` 

## T52 两个链表第一个公共节点

如果两个链表有公共节点，假设非公共部分长为 a, b, 公共部分长为 c, 用两个指针分别指向两个链表开头，每次往后走一步，若到表尾则回到另一个表的开头，则它们会在第一个公共节点处相遇，此时路程为 a + b + c, 如果回到表头的总次数超过 2，说明没有公共点

```java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode(int x) {
 *         val = x;
 *         next = null;
 *     }
 * }
 */
public class Solution {
    public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
        if(headA == null || headB == null)return null;
        ListNode curA = headA, curB = headB;
        int cnt = 0;
        while(curA != null && curB != null){
            if(curA == curB)return curA;
            curA = curA.next;
            curB = curB.next;
            if(curA == null){
                curA = headB;
                cnt ++;
            }
            if(curB == null){
                curB = headA;
                cnt ++;
            }
            if(cnt > 2)return null;
        }

        return null;
    }
}

```

## T57 和为 s 的两个数字

法一 : 二分查找, 对于每个数 nums[i], 查找有没有 j > i ,使得 nums[j] = target - nums[i]

时间复杂度`O(nlogn)`

法二: 双指针

```java
class Solution {
    public int[] twoSum(int[] nums, int target) {
        int i = 0, j = nums.length-1;
        while(i < j){
            if(nums[i] + nums[j] == target)return new int[]{nums[i], nums[j]};
            else if(nums[i] + nums[j] > target)j--;
            else i ++;
        }

        return new int[0];
    }
}

```

## T58 翻转单词顺序

用单个空格 split 之后手动过滤 / 用正则表达式  split

```java
class Solution {
    public String reverseWords(String s) {
        String[] words = s.trim().split(" ");
        StringBuilder reverseStr = new StringBuilder();
        for(int i = words.length-1; i >= 0; i --){
            if("".equals(words[i].trim()))continue;
            reverseStr.append(words[i]);
            if(i > 0)reverseStr.append(" ");
        }

        return reverseStr.toString();
    }
}

```

```java
class Solution {
    public String reverseWords(String s) {
        String[] words = s.trim().split("\\s+");
        StringBuilder reverseStr = new StringBuilder();
        for(int i = words.length-1; i >= 0; i --){
            //if("".equals(words[i].trim()))continue;
            reverseStr.append(words[i]);
            if(i > 0)reverseStr.append(" ");
        }

        return reverseStr.toString();
    }
}

```

## T45 把数组排成最小的数

贪心

(要排序首先要确定给出的标准能给出一个全序关系)

```java
class Solution {
    public String minNumber(int[] nums) {
        StringBuilder minNum = new StringBuilder();
        List<String> lst = new ArrayList<>();
        for(int i = 0; i < nums.length; i ++){
            lst.add(String.valueOf(nums[i]));
        }

        Collections.sort(lst, (s1, s2)->(s1+s2).compareTo(s2+s1));
        for(int i = 0; i < lst.size(); i ++){
            minNum.append(lst.get(i));
        }

        return minNum.toString();
    }
}
```

## T61 扑克牌中的顺子

排序之后按规则模拟，先统计 0 的个数。

```java
class Solution {
    public boolean isStraight(int[] nums) {
        int len = nums.length;
        if(len != 5)return false;
        Arrays.sort(nums);
        int countZero = 0;
        while(countZero < len && nums[countZero] == 0)countZero ++;
        int prev = -1;
        for(int i = countZero; i < len; i ++){
            if(prev == -1)prev = nums[i];
            else {
                if(prev+1 == nums[i])prev = nums[i];
                else {
                    while(prev+1 != nums[i] && countZero >= 0){
                        prev = prev + 1;
                        countZero --;
                    }
                    if(countZero < 0)return false;
                    prev = nums[i];
                }
            }
        }

        return true;
    }
}

```

## T40 最小的 k 个数

维护一个大根堆，堆元素个数超过 k 就弹出堆顶，扫描一趟数组之后堆里就是最小的 k 个数。

```java
class Solution {
    public int[] getLeastNumbers(int[] arr, int k) {
        PriorityQueue<Integer> pq = new PriorityQueue<>(k+1, (i1, i2)->-i1.compareTo(i2));
        for(int i = 0; i < arr.length; i ++){
            pq.offer(arr[i]);
            if(pq.size() > k)pq.poll();
        }

        int[] res = new int[k];
        for(int i = k-1; i >=0 ; i --){
            res[i] = pq.poll();
        }

        return res;
    }
}

```

## T68 二叉搜索树的最近公共祖先

根据二叉搜索树的性质，如果 p.val 和 q.val 都同时大于或者小于 root.val, 那么他们的 lca 一定在左右子树之一中，否则, p, q 属于不同子树, lca 是 root. 如果 p 或者 q 本身就是 root, 那么它本身就是 lca

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
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        if(root.val == p.val || root.val == q.val){
            return root.val == p.val ? p : q;
        }   

        if(Math.max(p.val, q.val) < root.val)
            return lowestCommonAncestor(root.left, p, q);
        else if(Math.min(p.val, q.val) > root.val)
            return lowestCommonAncestor(root.right, p, q);
        return root;
    }
}
```

## T68 二叉树的最近公共祖先

扩充函数的定义,

1. 如果在root为根的树中，同时具有pq两个节点，则返回最近祖先
2. 如果在root为根中的树中，只有其中一个节点，则返回含有的那个节点
3. 如果两个节点都不存在，则返回null

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
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        if(root == null || root == p || root == q)return root;

        TreeNode l = lowestCommonAncestor(root.left, p, q);
        TreeNode r = lowestCommonAncestor(root.right, p, q);

        if(l != null && r != null)return root;

        if(l != null)return l;

        return r;
    }
}

```

## T7 重建二叉树

递归地重建左右子树，然后和根拼起来

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
    public TreeNode buildTree(int[] preorder, int[] inorder) {
        int len = preorder.length;

        return build(preorder, inorder, 0, len-1, 0, len-1);
    }

    private TreeNode build(int[] preorder, int[] inorder, int l1, int r1, int l2, int r2){
        if(l1 > r1)return null;
        if(l1 == r1)return new TreeNode(preorder[l1]);

        TreeNode root = new TreeNode(preorder[l1]);
        int index = l2;
        while(index <= r2 && inorder[index] != root.val)index ++;
        int leftSize = index - l2;
        TreeNode leftSub = build(preorder, inorder, l1+1, l1 + leftSize, l2, index-1);
        TreeNode rightSub = build(preorder, inorder, l1 + leftSize + 1, r1, index + 1, r2);

        root.left = leftSub;
        root.right = rightSub;

        return root;
    }
}

```

## T16 数值的整数次方

快速幂板子题

```java
class Solution {
    public double myPow(double x, int n) {
        return n > 0 ? qPow(x, n) : 1 / qPow(x, -(long)n);
    }

    public double qPow(double x, long n){
        double res = 1;
        while(n > 0){
            if((n&1) == 1)res = res * x;
            x = x * x;
            n >>= 1;
        }

        return res;
    }
}

```

## T33 二叉搜索树的后序遍历序列

先判断满不满足基本偏序性质（和根的大小关系），再递归判断子树是不是二叉搜索树

```java
class Solution {
    public boolean verifyPostorder(int[] postorder) {
        int len = postorder.length;

        return check(postorder, 0, len-1);
    }

    private boolean check(int[] postorder, int l, int r){
        if(l >= r)return true;

        int index = r - 1;
        while(index >= l && postorder[index] > postorder[r])index --;
        int tmp = index;
        while(tmp >= l && postorder[tmp] < postorder[r])tmp --;
        if(tmp >= l)return false;

        return check(postorder, l, index) && check(postorder, index+1, r-1);
    }
}

```

## T10 斐波那契数列

简单递推

```java
class Solution {
    public final int MOD = 1000000007;
    private int[] f;
    public int fib(int n) {
        if(n == 0)return 0;
        if(n == 1)return 1;
        f = new int[n+1];
        f[0] = 0;
        f[1] = 1;
        for(int i = 2; i <= n; i ++)f[i] = (f[i-1] + f[i-2] ) % MOD;

        return f[n];
    }
}

```

## T10 青蛙跳台阶

和斐波那契数列其实是一样的,要跳到 n 级，可以从 n-1 或者 n-2 跳上来

```java
class Solution {
    public final int MOD = 1000000007;
    private int[] f;
    public int numWays(int n) {
        if(n == 0 || n == 1)return 1;
        f = new int[n+1];
        f[0] = f[1] = 1;
        for(int i = 2; i <= n; i ++)f[i] = (f[i-1] + f[i-2]) % MOD;

        return f[n];
    }
}

```

## T63 股票的最大利润

枚举每个可能卖出的时间，在之前的最低价格买入

```java
class Solution {
    public int maxProfit(int[] prices) {
        if(prices.length < 2)return 0;
        int ans = 0;
        int minv = prices[0];
        for(int i = 1; i < prices.length; i ++){
            ans = Math.max(prices[i]-minv, ans);
            minv = Math.min(minv, prices[i]);
        }

        return ans;
    }
}

```

## T29 顺时针打印矩阵

硬模拟

```java
class Solution {
    private int[] dx = {0, 1, 0, -1};
    private int[] dy = {1, 0, -1, 0};
    private boolean[][] st;
    public int[] spiralOrder(int[][] matrix) {
        int rows = matrix.length;
        if(rows == 0)return new int[0];
        int cols = matrix[0].length;
        if(cols == 0)return new int[0];
        int[] res = new int[rows * cols];
        int count = 0;
        int dir = 0;
        st = new boolean[rows][cols];
        for(int i = 0; i < rows; i ++)
            for(int j = 0; j < cols; j ++)
                st[i][j] = false;
        int x = 0, y = 0;
        while(count < res.length){
            res[count++] = matrix[x][y];
            st[x][y] = true;
            int x_ = x + dx[dir], y_ = y + dy[dir];
            if(x_<0 || x_ >= rows || y_ < 0 || y_ >= cols || st[x_][y_])dir = (dir + 1) % 4;
            x = x + dx[dir];
            y = y + dy[dir];
        }

        return res;
    }
}

```

## T31 栈的压入弹出序列

模拟，每次压入一个元素之后看看能不能弹掉一些，最后如果能弹空就是合法序列

```java
class Solution {
    public boolean validateStackSequences(int[] pushed, int[] popped) {
        Deque<Integer> stk = new LinkedList<>();
        int len = pushed.length;
        if(len < 2)return true;
        int popIndex = 0;
        for(int i = 0; i < len; i ++){
            stk.push(pushed[i]);
            while(!stk.isEmpty() && popIndex < len && stk.peek() == popped[popIndex]){
                stk.pop();
                popIndex ++;
            }
        }

        return stk.isEmpty();
    }
}

```

## T42 连续子数组的最大和

简单 DP

```java
class Solution {
    private int[] f;
    public int maxSubArray(int[] nums) {
        int len = nums.length;
        f = new int[len];
        int ans = nums[0];
        f[0] = nums[0];
        for(int i = 1; i < len; i ++){
            f[i] = Math.max(f[i-1] + nums[i], nums[i]);
            ans = Math.max(ans, f[i]);
        }

        return ans;
    }
}

```

## T47 礼物的最大价值

定义 `f[i][j]` 为从左上角出发，走到 `(i, j)` 能获得的最大价值, `f[i][j] = grid[i][j]`, 对于每个位置，可以从它的上面或者左边到达，因此 `f[i][j] = max(f[i-1][j], f[i][j-1]) + grid[i][j]`

```java
class Solution {
    private int[][] f;
    public int maxValue(int[][] grid) {
        int rows = grid.length;
        int cols = grid[0].length;
        f = new int[rows][cols];
        f[0][0] = grid[0][0];
        for(int i = 1; i < cols; i ++)f[0][i] = f[0][i-1] + grid[0][i];
        for(int i = 1; i < rows; i ++)f[i][0] = f[i-1][0] + grid[i][0];
        for(int i = 1; i < rows; i ++)
            for(int j = 1; j < cols; j ++)
                f[i][j] = Math.max(f[i-1][j], f[i][j-1]) + grid[i][j];

        return f[rows-1][cols-1];        
    }
}

```

## T46 把数字翻译成字符串

base case : 对于数字 0~9, 只有一种翻译方法, 对于两位数 10 ~ 25, 有两种翻译方法。
对于数字 abcd...efg, 可能数字 ab 作为翻译的第一个字符，或者数字 a 作为翻译的第一个字符。
但是数字 ab 能够翻译成单个字符的条件是 ab > 9 && ab < 26, 根据以上可以写出递推式,这里使用记忆化搜索的写法.

```java
class Solution {
    private int[] f;
    public int translateNum(int num) {
        if(num < 10)return 1;
        List<Integer> lst = new ArrayList<>();
        while(num > 0){
            lst.add(num % 10);
            num /= 10;
        }
        f = new int[lst.size()];
        for(int i = 0; i < lst.size(); i ++)f[i] = -1;

        return count(0, lst.size()-1, lst);
    }

    private int count(int low, int high, List<Integer> lst){
        if(f[low] != -1)return f[low];
        if(low == high){
            f[low] = 1;
            return f[low];
        }
        if(high - low == 1){
            f[low] = 1;
            int part = lst.get(low) + lst.get(high) * 10;
            if(part > 9 && part < 26)f[low] += 1;
            return f[low];
        }

        int part = lst.get(low) + lst.get(low+1) * 10;
        f[low] = 0;
        f[low] += count(low+1, high, lst);
        if(part > 9 && part < 26)f[low] += count(low+2, high, lst);

        return f[low];
    }
}

```

