

(define (problem BW-rand-8)
(:domain blocksworld-4ops)
(:objects a b c d e f g h )
(:init
(handempty)
(on a d)
(on b c)
(on c g)
(on d e)
(on e h)
(ontable f)
(on g a)
(on h f)
(clear b)
)
(:goal
(and
(on b c)
(on c h)
(on d e)
(on e f)
(on g d))
)
)


