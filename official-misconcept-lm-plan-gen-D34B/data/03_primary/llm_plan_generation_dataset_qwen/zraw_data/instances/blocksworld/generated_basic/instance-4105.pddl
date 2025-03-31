

(define (problem BW-rand-5)
(:domain blocksworld-4ops)
(:objects a b c d e )
(:init
(handempty)
(ontable a)
(ontable b)
(on c b)
(on d e)
(ontable e)
(clear a)
(clear c)
(clear d)
)
(:goal
(and
(on c b)
(on d c)
(on e a))
)
)


