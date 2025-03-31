

(define (problem BW-rand-9)
(:domain blocksworld-4ops)
(:objects a b c d e f g h i )
(:init
(handempty)
(on a f)
(ontable b)
(on c i)
(ontable d)
(ontable e)
(on f h)
(on g a)
(on h d)
(on i b)
(clear c)
(clear e)
(clear g)
)
(:goal
(and
(on a d)
(on b c)
(on c i)
(on d b)
(on e g)
(on f a)
(on g f)
(on h e))
)
)


