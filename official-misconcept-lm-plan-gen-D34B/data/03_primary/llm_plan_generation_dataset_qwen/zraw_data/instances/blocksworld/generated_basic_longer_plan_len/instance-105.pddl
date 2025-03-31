

(define (problem BW-rand-8)
(:domain blocksworld-4ops)
(:objects a b c d e f g h )
(:init
(handempty)
(on a g)
(on b a)
(on c e)
(on d h)
(on e f)
(on f b)
(on g d)
(ontable h)
(clear c)
)
(:goal
(and
(on b g)
(on c a)
(on f c)
(on g d)
(on h f))
)
)


