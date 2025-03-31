

(define (problem BW-rand-10)
(:domain blocksworld-4ops)
(:objects a b c d e f g h i j )
(:init
(handempty)
(on a i)
(on b g)
(on c b)
(on d j)
(on e a)
(ontable f)
(on g d)
(on h e)
(on i f)
(on j h)
(clear c)
)
(:goal
(and
(on a f)
(on b i)
(on c j)
(on d h)
(on e b)
(on f d)
(on i g)
(on j e))
)
)


