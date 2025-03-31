

(define (problem BW-rand-6)
(:domain blocksworld-4ops)
(:objects a b c d e f )
(:init
(handempty)
(on a d)
(on b e)
(on c f)
(on d b)
(on e c)
(ontable f)
(clear a)
)
(:goal
(and
(on a d)
(on d b)
(on e c)
(on f e))
)
)


