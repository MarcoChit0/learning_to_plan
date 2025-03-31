

(define (problem BW-rand-9)
(:domain blocksworld-4ops)
(:objects a b c d e f g h i )
(:init
(handempty)
(on a g)
(ontable b)
(on c b)
(on d a)
(on e d)
(on f c)
(ontable g)
(on h f)
(on i h)
(clear e)
(clear i)
)
(:goal
(and
(on a e)
(on b c)
(on c d)
(on d a)
(on e g)
(on g h)
(on h i)
(on i f))
)
)


