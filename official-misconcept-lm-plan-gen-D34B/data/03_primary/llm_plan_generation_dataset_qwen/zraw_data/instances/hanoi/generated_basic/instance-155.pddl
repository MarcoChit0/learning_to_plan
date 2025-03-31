


(define (problem hanoi-4)
(:domain hanoi)
(:objects peg1 peg2 peg3 d345 d337 d49 d224 )
(:init
(smaller peg1 d345)
(smaller peg1 d337)
(smaller peg1 d49)
(smaller peg1 d224)
(smaller peg2 d345)
(smaller peg2 d337)
(smaller peg2 d49)
(smaller peg2 d224)
(smaller peg3 d345)
(smaller peg3 d337)
(smaller peg3 d49)
(smaller peg3 d224)
(smaller d337 d345)
(smaller d49 d345)
(smaller d224 d345)
(smaller d49 d337)
(smaller d224 d337)
(smaller d224 d49)
(clear peg2)
(clear peg3)
(clear d345)
(on d224 peg1)
(on d49 d224)
(on d337 d49)
(on d345 d337)
)
(:goal
(and 
(on d224 peg3)
(on d49 d224)
(on d337 d49)
(on d345 d337)
)
)
)


