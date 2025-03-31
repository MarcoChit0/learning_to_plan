

(define (problem BW-rand-9)
(:domain blocksworld-4ops)
(:objects a b c d e f g h i )
(:init
(handempty)
(on a i)
(on b c)
(on c d)
(on d f)
(on e a)
(ontable f)
(ontable g)
(on h b)
(on i h)
(clear e)
(clear g)
)
(:goal
(and
(on a e)
(on b f)
(on c h)
(on d a)
(on e b)
(on f i)
(on g d)
(on h g))
)
)


