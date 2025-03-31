

(define (problem BW-rand-6)
(:domain blocksworld-4ops)
(:objects a b c d e f )
(:init
(handempty)
(on a d)
(on b c)
(on c a)
(on d f)
(ontable e)
(ontable f)
(clear b)
(clear e)
)
(:goal
(and
(on a d)
(on b e)
(on c f)
(on d c)
(on e a))
)
)


