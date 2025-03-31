


(define (problem hanoi-4)
(:domain hanoi)
(:objects peg1 peg2 peg3 d448 d324 d247 d441 )
(:init
(smaller peg1 d448)
(smaller peg1 d324)
(smaller peg1 d247)
(smaller peg1 d441)
(smaller peg2 d448)
(smaller peg2 d324)
(smaller peg2 d247)
(smaller peg2 d441)
(smaller peg3 d448)
(smaller peg3 d324)
(smaller peg3 d247)
(smaller peg3 d441)
(smaller d324 d448)
(smaller d247 d448)
(smaller d441 d448)
(smaller d247 d324)
(smaller d441 d324)
(smaller d441 d247)
(clear peg2)
(clear peg3)
(clear d448)
(on d441 peg1)
(on d247 d441)
(on d324 d247)
(on d448 d324)
)
(:goal
(and 
(on d441 peg3)
(on d247 d441)
(on d324 d247)
(on d448 d324)
)
)
)


