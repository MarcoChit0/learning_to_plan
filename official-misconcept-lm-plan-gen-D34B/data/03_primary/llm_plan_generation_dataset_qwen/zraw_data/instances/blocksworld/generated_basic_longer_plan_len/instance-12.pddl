

(define (problem BW-rand-6)
(:domain blocksworld-4ops)
(:objects a b c d e f )
(:init
(handempty)
(on a b)
(on b e)
(on c d)
(on d a)
(ontable e)
(on f c)
(clear f)
)
(:goal
(and
(on a b)
(on b f)
(on c d)
(on d e)
(on f c))
)
)


