

(define (problem BW-rand-6)
(:domain blocksworld-4ops)
(:objects a b c d e f )
(:init
(handempty)
(ontable a)
(on b e)
(on c a)
(on d c)
(on e d)
(on f b)
(clear f)
)
(:goal
(and
(on a e)
(on b c)
(on c a)
(on f b))
)
)


