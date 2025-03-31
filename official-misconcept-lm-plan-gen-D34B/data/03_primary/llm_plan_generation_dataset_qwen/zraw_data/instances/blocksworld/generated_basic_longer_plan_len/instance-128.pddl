

(define (problem BW-rand-9)
(:domain blocksworld-4ops)
(:objects a b c d e f g h i )
(:init
(handempty)
(on a e)
(on b f)
(on c b)
(on d i)
(on e g)
(on f a)
(on g d)
(ontable h)
(ontable i)
(clear c)
(clear h)
)
(:goal
(and
(on a c)
(on b i)
(on c f)
(on d g)
(on e d)
(on f h)
(on h b)
(on i e))
)
)


