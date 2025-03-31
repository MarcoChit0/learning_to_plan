

(define (problem BW-rand-9)
(:domain blocksworld-4ops)
(:objects a b c d e f g h i )
(:init
(handempty)
(on a i)
(on b d)
(on c g)
(ontable d)
(on e a)
(on f h)
(on g f)
(on h b)
(on i c)
(clear e)
)
(:goal
(and
(on a b)
(on b d)
(on c i)
(on d e)
(on e g)
(on g f)
(on h a)
(on i h))
)
)


