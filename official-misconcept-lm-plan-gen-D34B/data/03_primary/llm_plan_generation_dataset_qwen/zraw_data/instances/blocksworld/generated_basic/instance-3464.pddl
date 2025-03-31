

(define (problem BW-rand-5)
(:domain blocksworld-4ops)
(:objects a b c d e )
(:init
(handempty)
(ontable a)
(on b a)
(on c e)
(ontable d)
(on e b)
(clear c)
(clear d)
)
(:goal
(and
(on a b)
(on b c)
(on c d)
(on d e))
)
)


