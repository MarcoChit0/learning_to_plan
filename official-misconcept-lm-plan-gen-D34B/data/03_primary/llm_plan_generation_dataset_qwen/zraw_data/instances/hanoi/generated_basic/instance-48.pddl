


(define (problem hanoi-1)
(:domain hanoi)
(:objects peg1 peg2 peg3 d134 )
(:init
(smaller peg1 d134)
(smaller peg2 d134)
(smaller peg3 d134)
(clear peg2)
(clear peg3)
(clear d134)
(on d134 peg1)
)
(:goal
(and 
(on d134 peg3)
)
)
)


