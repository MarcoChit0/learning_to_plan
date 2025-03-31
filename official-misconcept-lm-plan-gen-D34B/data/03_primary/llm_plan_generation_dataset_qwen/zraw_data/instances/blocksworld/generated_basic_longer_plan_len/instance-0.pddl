

(define (problem BW-rand-6)
(:domain blocksworld-4ops)
(:objects a b c d e f )
(:init
(handempty)
(ontable a)
(on b d)
(on c a)
(on d c)
(on e f)
(on f b)
(clear e)
)
(:goal
(and
(on a b)
(on d f)
(on e d)
(on f c))
)
)


