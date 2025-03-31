

(define (problem BW-rand-8)
(:domain blocksworld-4ops)
(:objects a b c d e f g h )
(:init
(handempty)
(on a b)
(on b g)
(ontable c)
(ontable d)
(on e a)
(on f d)
(on g f)
(on h e)
(clear c)
(clear h)
)
(:goal
(and
(on a d)
(on b g)
(on c e)
(on d b)
(on e f)
(on f a)
(on h c))
)
)


