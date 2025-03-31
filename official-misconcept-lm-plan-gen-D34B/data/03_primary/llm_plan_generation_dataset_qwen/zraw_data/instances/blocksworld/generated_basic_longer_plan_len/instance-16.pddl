

(define (problem BW-rand-6)
(:domain blocksworld-4ops)
(:objects a b c d e f )
(:init
(handempty)
(ontable a)
(on b c)
(on c d)
(on d f)
(ontable e)
(on f e)
(clear a)
(clear b)
)
(:goal
(and
(on a b)
(on b d)
(on c e)
(on e a)
(on f c))
)
)


