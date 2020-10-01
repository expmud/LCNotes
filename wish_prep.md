# 0002 Add Two Numbers
## Linked lists
```    
def addTwoNumbers(l1, l2):
    cur = head = ListNode(0)
    p = l1
    q = l2
    carry = 0
    while q != None or p != None:
        x = 0 if p == None else p.val
        y = 0 if q == None else q.val
        s = x + y + carry
        carry = s / 10
        cur.next = ListNode(s % 10)
        cur = cur.next
        if p != None:
            p = p.next
        if q != None:
            q = q.next
    if carry > 0:
        cur.next = ListNode(carry)
    
    return head.nex
````
# 0085 Longest Mountain in Array
## Two Pointers
```
def longestMountain(A):
    s = len(A)
    base = ans = 0
    
    while base < s:
        end = base
        if end + 1 < s and A[end] < A[end + 1]:
            while end + 1 < s and A[end] < A[end + 1]:
                end += 1
            if end + 1 < s and A[end] > A[end + 1]:
                while end + 1 < s and A[end] > A[end + 1]:
                    end += 1
                ans  = max(ans, end - base + 1)
    
        base = max(end, base + 1)
    return ans
```

# 0528 Random Pick with Weight
## Binary Search
```
class Solution:
    def __init__(self, w: List[int]):
        """
        :type w: List[int]
        """
        self.prefix_sums = []
        prefix_sum = 0
        for weight in w:
            prefix_sum += weight
            self.prefix_sums.append(prefix_sum)
        self.total_sum = prefix_sum

    def pickIndex(self) -> int:
        """
        :rtype: int
        """
        target = self.total_sum * random.random()
        # run a binary search to find the target zone
        low, high = 0, len(self.prefix_sums)
        while low < high:
            mid = low + (high - low) // 2
            if target > self.prefix_sums[mid]:
                low = mid + 1
            else:
                high = mid
        return low
```