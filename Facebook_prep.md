# 0023 Merge k Sorted Lists
Topic: Divide and Conquer, Linked List

## Brute Force
```
def merge_k_sorted_lists(lists):
    head = point = ListNode(0)
    arr = []
    for lst in Lists:
        while lst:
            arr.append(lst.val)
            lst = lst.next
    for x in sorted(arr):
        point.next = ListNode(x)
        point = point.next
    return head.next
```
Time Complexity: O(N log N), N is total number of nodes
- collecting nodes: O(N)
- sorting: O(N log N)
- putting together: O(N)

## D&C
```
def merge(s1, s2):
    head = ret = ListNode(0)
    while(1):
        if (s1 == None):
            ret.next = s2
            return head.next
        if (s2 == None):
            ret.next = s1
            return head.next
        if (s1.val <= s2.val):
            ret.next = s1
            s1 = s1.next
        else:
            ret.next = s2
            s2 = s2.next
        ret = ret.next
    return head.next
    
def mergeKLists(lists):
    """
    :type lists: List[ListNode]
    :rtype: ListNode
    """
    
    amount = len(lists)
    interval = 1
    while interval < amount:
        for i in range(0, amount - interval, interval * 2):
            lists[i] = erge(lists[i], lists[i + interval])
        interval *= 2
    return lists[0] if amount > 0 else None
```
Time complextiy: O(N log k) where N is total number of nodes and k is number of linked lists
- merge to lists in O(n) as n is the number of nodes in both of the lists
- number of calls to merge is O(log k)

# 0034 Find First and Last Position of Element in Sorted Array
Topic: Binary Search
## Brute Force
```
def searchRange(nums, target):
    for i in ragne(len(nums)):
        if (nums[i] == target):
            left_id = i
            break
    else:
        return [-1, -1]
    
    for j in range(len(nums)-1, -1, -1):
        if (nums[i] == target):
            right_id = i
            break
    return [left_id, right_id]
```
Time complexity: O(N)

## Binary Search
```
def BSearch(nums, target, left):
    l = 0
    r = len(nums)
    while l < r:
        mid = (l + r) // 2
        if (nums[mid] > target or (left and nums[mid] == target)):
            r = mid
        else:
            l = mid + 1
    return l

def searchRange(nums, target):
    left = BSearch(nums, target, True)

    if left == len(nums) or nums[left] != target:
        return [-1, -1]
    
    right = BSearch(nums, target, False) - 1
    return [left, right]
```
Tiem Complexity: O(Log N)

# 0056 Merge Intervals
Sorting
## Sort
```
def merge(intervals):
    merged = []
    intervals.sort(key=lambda x:x[0])
    
    for interval in intervals:
        if not merged or merged[-1][1] < interval[0]:
            merged.append(interval)
        else:
            merged[-1][1] = max(interval[1], merged[-1][1])
    return merged
```
Time complexity: O(n log n)

# 0067 Add Binary
Topic: ath, String, Bit manipulation
## Casting
```
def addBinary(a, b):
    return '{0:b}'.format(int(a, 2) + int(b, 2))
```

## Bit Shift
```
def addBinary(a, b)
    x, y = int(a, 2), int(b, 2)
    while y:
        answer = x ^ y
        carry = (x & y) << 1
        x, y = answer, carry
    return bin(x)[2:]
```
Time complexity: O(N + M), N and M are length of a and b

# Merge Sorted Array
Topic: two pointers
## Brute Force(Merge and Sort)
```
def merge(nums1, m, nums2, n):
    nums1[:] = sorted(nums1[:m] + nums2)
```
Time complexity: O((N+M) log (N+M))
## Tow pointers
```
def merge(nums1, m, nums2, n):
    p1 = m - 1
    p2 = n - 1
    p = m + n - 1
    while p1 >= 0 and p2 >= 0:
        if nums1[p1] < nums2[p2]:
            nums1[p] = nums2[p2]
            p2 -= 1
        else:
            nums1[p] = nums1[p1]
            p1 -= 1
        p -= 1
    nums1[:p2 + 1] = nums2[:p2 + 1]
```
Time complexity: O(N+M)

# 0124 Binary Tree Maximum Path Sum
Topic: DFS
## DFS 
```
def maxPathSum(root):
    def max_gain(node):
        nonlocal max_sum
        if not node:
            return 0

        # max sum on the left and right sub-trees of node
        left_gain = max(max_gain(node.left), 0)
        right_gain = max(max_gain(node.right), 0)
        
        # the price to start a new path where `node` is a highest node
        price_newpath = node.val + left_gain + right_gain
        
        # update max_sum if it's better to start a new path
        max_sum = max(max_sum, price_newpath)
    
        # for recursion :
        # return the max gain if continue the same path
        return node.val + max(left_gain, right_gain)

    max_sum = float('-inf')
    max_gain(root)
    return max_sum
```
Time complexity: O(N), n is number of nodes

# 0125 Valid Palindrome
Topic: Two Pointer
## Two Pointer
```
def isPalindrome(s):
    l, r = 0, len(s) - 1
    while l < r:
        while l < r and not s[l].isalnum():
            l += 1
        while l < r and not s[r].isalnum():
            r -= 1
        
        if l < r and s[l].lower() != s[r].lower():
            return False
        
        l += 1
        r -= 1
    return True
```
Time complexity: O(n)

# 0133 Cloen Graph
## DFS 
```
class Solution:
    def __init__(self):
        self.visited = {}
        
    def cloneGraph(self, node: 'Node') -> 'Node':
        if not node:
            return node
        if node in self.visited:
            return self.visited[node]
        
        tmp = Node(node.val, [])
        
        self.visited[node] = tmp
        
        if node.neighbors:
            tmp.neighbors = [self.cloneGraph(n) for n in node.neighbors]
            
        return tmp
```
Time complexity: O(N)

# 0173 Binary Search Tree Iterator
## Stack + DFS
```
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class BSTIterator:

    def __init__(self, root: TreeNode):
        self.stack = []
        
        self.get_left_most_node(root)
    
    def get_left_most_node(self, node):
        while node:
            self.stack.append(node)
            node = node.left
            
    def next(self) -> int:
        """
        @return the next smallest number
        """
        node = self.stack.pop()
        
        if node.right:
            self.get_left_most_node(node.right)
        return node.val
        

    def hasNext(self) -> bool:
        """
        @return whether we have a next smallest number
        """
        return len(self.stack) > 0
```
# 0238 Product of Array Except Self
## Array Manipulation
```
def productExceptSelf(nums: List[int]) -> List[int]:
    size = len(nums)
    answer = [0] * size
    answer[0] = 1
    for i in range(1, size):
        answer[i] = nums[i-1] * answer[i-1]
    
    R = 1
    for i in reversed(range(size)):
        answer[i] = R * answer[i]
        R *= nums[i]
    return answer
```

# 0953 Verifying an Alien Dictionary
## Hash Table
```
def isAlienSorted(words: List[str], order: str) -> bool:
    alph_index = {c:i for i, c in enumerate(order)}
    
    for i in range(len(words) - 1):
        word1 = words[i]
        word2 = words[i + 1]
        
        for j in range(min(len(word1), len(word2))):
            if word1[j] != word2[j]:
                if alph_index[word1[j]] > alph_index[word2[j]]:
                    return False
                break
        else:
            if len(word1) > len(word2):
                return False
            
    return True
```
Time complexity: O(c)

# 0273 Integer to English Words
## Divide and Conquer
```
class Solution:
    def numberToWords(self, num):
        """
        :type num: int
        :rtype: str
        """
        def one(num):
            switcher = {
                1: 'One',
                2: 'Two',
                3: 'Three',
                4: 'Four',
                5: 'Five',
                6: 'Six',
                7: 'Seven',
                8: 'Eight',
                9: 'Nine'
            }
            return switcher.get(num)

        def two_less_20(num):
            switcher = {
                10: 'Ten',
                11: 'Eleven',
                12: 'Twelve',
                13: 'Thirteen',
                14: 'Fourteen',
                15: 'Fifteen',
                16: 'Sixteen',
                17: 'Seventeen',
                18: 'Eighteen',
                19: 'Nineteen'
            }
            return switcher.get(num)
        
        def ten(num):
            switcher = {
                2: 'Twenty',
                3: 'Thirty',
                4: 'Forty',
                5: 'Fifty',
                6: 'Sixty',
                7: 'Seventy',
                8: 'Eighty',
                9: 'Ninety'
            }
            return switcher.get(num)
        

        def two(num):
            if not num:
                return ''
            elif num < 10:
                return one(num)
            elif num < 20:
                return two_less_20(num)
            else:
                tenner = num // 10
                rest = num - tenner * 10
                return ten(tenner) + ' ' + one(rest) if rest else ten(tenner)
        
        def three(num):
            hundred = num // 100
            rest = num - hundred * 100
            if hundred and rest:
                return one(hundred) + ' Hundred ' + two(rest) 
            elif not hundred and rest: 
                return two(rest)
            elif hundred and not rest:
                return one(hundred) + ' Hundred'
        
        billion = num // 1000000000
        million = (num - billion * 1000000000) // 1000000
        thousand = (num - billion * 1000000000 - million * 1000000) // 1000
        rest = num - billion * 1000000000 - million * 1000000 - thousand * 1000
        
        if not num:
            return 'Zero'
        
        result = ''
        if billion:        
            result = three(billion) + ' Billion'
        if million:
            result += ' ' if result else ''    
            result += three(million) + ' Million'
        if thousand:
            result += ' ' if result else ''
            result += three(thousand) + ' Thousand'
        if rest:
            result += ' ' if result else ''
            result += three(rest)
        return result
```

# 0973 K Closest Points to Origin
## Divide and Conquer
```
def kClosest(points, K):
    dist = lambda i: points[i][0]**2 + points[i][1]**2
    def sort(i, j, K):
        # Partially sorts A[i:j+1] so the first K elements are
        # the smallest K elements.
        if i >= j: return

        # Put random element as A[i] - this is the pivot
        k = random.randint(i, j)
        points[i], points[k] = points[k], points[i]

        mid = partition(i, j)
        if K < mid - i + 1:
            sort(i, mid - 1, K)
        elif K > mid - i + 1:
            sort(mid + 1, j, K - (mid - i + 1))

    def partition(i, j):
        # Partition by pivot A[i], returning an index mid
        # such that A[i] <= A[mid] <= A[j] for i < mid < j.
        oi = i
        pivot = dist(i)
        i += 1

        while True:
            while i < j and dist(i) < pivot:
                i += 1
            while i <= j and dist(j) >= pivot:
                j -= 1
            if i >= j: break
            points[i], points[j] = points[j], points[i]

        points[oi], points[j] = points[j], points[oi]
        return j
        
    sort(0, len(points) - 1, K)
    return points[:K]
```

# 0496 Next Greater Element I
```
public class Solution {
    public int[] nextGreaterElement(int[] findNums, int[] nums) {
        Stack < Integer > stack = new Stack < > ();
        HashMap < Integer, Integer > map = new HashMap < > ();
        int[] res = new int[findNums.length];
        for (int i = 0; i < nums.length; i++) {
            while (!stack.empty() && nums[i] > stack.peek())
                map.put(stack.pop(), nums[i]);
            stack.push(nums[i]);
        }
        while (!stack.empty())
            map.put(stack.pop(), -1);
        for (int i = 0; i < findNums.length; i++) {
            res[i] = map.get(findNums[i]);
        }
        return res;
    }
}
```

# 1027. Longest Arithmetic Subsequence
```
class Solution:
    def longestArithSeqLength(self, A):
        n=len(A)
        dp={}
        for i in range(n):
            for j in range(i+1,n):
                dif = A[j]-A[i]
                if (i,dif) in dp :
                    dp[(j,dif)]=dp[(i,dif)]+1
                else:
                    dp[(j,dif)]=2
        return max(dp.values())
```


# 958. Check Completeness of a Binary Tree
```
class Solution(object):
    def isCompleteTree(self, root):
        nodes = [(root, 1)]
        i = 0
        while i < len(nodes):
            node, v = nodes[i]
            i += 1
            if node:
                nodes.append((node.left, 2*v))
                nodes.append((node.right, 2*v+1))

        return  nodes[-1][1] == len(nodes)
```

```
class Solution:
    def isCompleteTree(self, root: TreeNode) -> bool:
        arr1 = []
        arr2 = []
        arr1.append(root)
        x = 0
        while len(arr1) != 0:
            hasNoChild = False
            tmp = len(arr1)
            while len(arr1) != 0:
                node = arr1.pop()
                if node.left != None:
                    if hasNoChild:
                        return False
                    arr2.append(node.left)
                else:
                    hasNoChild = True
                    
                if node.right != None:
                    if hasNoChild:
                        return False
                    arr2.append(node.right)
                else:
                    hasNoChild = True
            arr2.reverse()
            arr1 = arr2
            arr2 = []
            print(x, tmp, len(arr1))
            if tmp < 2**x and len(arr1) != 0:
                return False
            x += 1
        return True
```