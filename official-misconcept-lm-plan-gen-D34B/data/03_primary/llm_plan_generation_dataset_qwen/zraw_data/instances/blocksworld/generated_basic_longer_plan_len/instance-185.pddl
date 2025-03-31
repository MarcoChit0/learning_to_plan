

(define (problem BW-rand-9)
(:domain blocksworld-4ops)
(:objects a b c d e f g h i )
(:init
(handempty)
(on a d)
(on b i)
(on c h)
(on d f)
(on e c)
(on f g)
(ontable g)
(on h b)
(on i a)
(clear e)
)
(:goal
(and
(on a c)
(on b i)
(on c d)
(on e f)
(on f a)
(on g b)
(on h e)
(on i h))
)
)


