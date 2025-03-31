

(define (problem BW-rand-9)
(:domain blocksworld-4ops)
(:objects a b c d e f g h i )
(:init
(handempty)
(on a c)
(on b f)
(on c d)
(on d e)
(on e h)
(on f g)
(on g a)
(on h i)
(ontable i)
(clear b)
)
(:goal
(and
(on a h)
(on b f)
(on d i)
(on e g)
(on g a)
(on i b))
)
)


