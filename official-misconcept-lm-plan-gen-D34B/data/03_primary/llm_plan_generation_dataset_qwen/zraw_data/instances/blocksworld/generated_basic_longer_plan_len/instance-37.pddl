

(define (problem BW-rand-6)
(:domain blocksworld-4ops)
(:objects a b c d e f )
(:init
(handempty)
(on a d)
(on b f)
(on c e)
(on d c)
(ontable e)
(ontable f)
(clear a)
(clear b)
)
(:goal
(and
(on a b)
(on b d)
(on d e)
(on e f)
(on f c))
)
)


