

(define (problem BW-rand-8)
(:domain blocksworld-4ops)
(:objects a b c d e f g h )
(:init
(handempty)
(on a h)
(on b f)
(on c a)
(on d e)
(on e b)
(on f c)
(ontable g)
(on h g)
(clear d)
)
(:goal
(and
(on a e)
(on b c)
(on d h)
(on e g)
(on f d)
(on g f))
)
)


