

(define (problem BW-rand-9)
(:domain blocksworld-4ops)
(:objects a b c d e f g h i )
(:init
(handempty)
(on a f)
(on b c)
(ontable c)
(on d g)
(on e a)
(on f i)
(on g b)
(on h d)
(on i h)
(clear e)
)
(:goal
(and
(on a e)
(on b f)
(on c a)
(on d g)
(on f i)
(on g h)
(on h b)
(on i c))
)
)


