

(define (problem BW-rand-9)
(:domain blocksworld-4ops)
(:objects a b c d e f g h i )
(:init
(handempty)
(on a f)
(on b i)
(on c b)
(on d a)
(on e d)
(on f g)
(on g c)
(ontable h)
(on i h)
(clear e)
)
(:goal
(and
(on a f)
(on b a)
(on c i)
(on d g)
(on e c)
(on g h)
(on h b)
(on i d))
)
)


