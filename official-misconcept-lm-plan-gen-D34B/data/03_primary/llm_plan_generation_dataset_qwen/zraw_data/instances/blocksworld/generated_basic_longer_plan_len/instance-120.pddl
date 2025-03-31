

(define (problem BW-rand-8)
(:domain blocksworld-4ops)
(:objects a b c d e f g h )
(:init
(handempty)
(on a c)
(on b f)
(on c e)
(on d g)
(ontable e)
(on f d)
(ontable g)
(ontable h)
(clear a)
(clear b)
(clear h)
)
(:goal
(and
(on a e)
(on b f)
(on c a)
(on d c)
(on e g)
(on g b)
(on h d))
)
)


