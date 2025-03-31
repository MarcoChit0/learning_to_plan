

(define (problem BW-rand-9)
(:domain blocksworld-4ops)
(:objects a b c d e f g h i )
(:init
(handempty)
(on a f)
(ontable b)
(on c d)
(ontable d)
(on e g)
(on f c)
(on g h)
(on h a)
(on i b)
(clear e)
(clear i)
)
(:goal
(and
(on a b)
(on b f)
(on c d)
(on d i)
(on e c)
(on f h)
(on g e)
(on h g))
)
)


