

(define (problem BW-rand-6)
(:domain blocksworld-4ops)
(:objects a b c d e f )
(:init
(handempty)
(on a c)
(on b f)
(on c d)
(on d e)
(ontable e)
(ontable f)
(clear a)
(clear b)
)
(:goal
(and
(on a b)
(on b f)
(on c d)
(on e a)
(on f c))
)
)


