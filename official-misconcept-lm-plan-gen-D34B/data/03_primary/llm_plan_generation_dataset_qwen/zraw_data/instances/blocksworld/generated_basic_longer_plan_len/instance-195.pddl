

(define (problem BW-rand-10)
(:domain blocksworld-4ops)
(:objects a b c d e f g h i j )
(:init
(handempty)
(on a d)
(on b i)
(on c h)
(on d b)
(on e j)
(ontable f)
(on g c)
(ontable h)
(on i e)
(on j f)
(clear a)
(clear g)
)
(:goal
(and
(on a c)
(on b j)
(on c i)
(on d h)
(on e g)
(on f e)
(on g d)
(on h a)
(on i b))
)
)


