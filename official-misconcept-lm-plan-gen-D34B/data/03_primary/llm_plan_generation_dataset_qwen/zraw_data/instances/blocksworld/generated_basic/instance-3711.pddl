

(define (problem BW-rand-5)
(:domain blocksworld-4ops)
(:objects a b c d e )
(:init
(handempty)
(on a d)
(on b e)
(on c a)
(ontable d)
(ontable e)
(clear b)
(clear c)
)
(:goal
(and
(on a d)
(on c e)
(on d b)
(on e a))
)
)


