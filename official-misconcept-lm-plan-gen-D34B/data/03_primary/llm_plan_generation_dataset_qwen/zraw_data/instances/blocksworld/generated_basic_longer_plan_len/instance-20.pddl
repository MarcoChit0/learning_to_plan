

(define (problem BW-rand-6)
(:domain blocksworld-4ops)
(:objects a b c d e f )
(:init
(handempty)
(on a f)
(on b a)
(on c b)
(on d e)
(ontable e)
(on f d)
(clear c)
)
(:goal
(and
(on a d)
(on b e)
(on c f)
(on e c))
)
)


