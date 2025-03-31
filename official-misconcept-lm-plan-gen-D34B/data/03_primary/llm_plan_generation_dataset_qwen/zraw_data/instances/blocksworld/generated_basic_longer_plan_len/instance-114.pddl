

(define (problem BW-rand-8)
(:domain blocksworld-4ops)
(:objects a b c d e f g h )
(:init
(handempty)
(ontable a)
(on b c)
(on c g)
(on d e)
(on e f)
(on f h)
(on g d)
(on h a)
(clear b)
)
(:goal
(and
(on b g)
(on c a)
(on d c)
(on e h)
(on g d)
(on h f))
)
)


