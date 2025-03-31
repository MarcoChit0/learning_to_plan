(define (problem prob)
 (:domain barman)
 (:objects 
      shaker126 - shaker
      left right - hand
      shot462 shot338 - shot
      ingredient169 ingredient283 ingredient346 ingredient327 - ingredient
      cocktail425 - cocktail
      dispenser342 dispenser7 dispenser194 dispenser245 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker126)
  (ontable shot462)
  (ontable shot338)
  (dispenses dispenser342 ingredient169)
  (dispenses dispenser7 ingredient283)
  (dispenses dispenser194 ingredient346)
  (dispenses dispenser245 ingredient327)
  (clean shaker126)
  (clean shot462)
  (clean shot338)
  (empty shaker126)
  (empty shot462)
  (empty shot338)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker126 l0)
  (shaker-level shaker126 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail425 ingredient346)
  (cocktail-part2 cocktail425 ingredient283)
)
 (:goal
  (and
      (contains shot462 cocktail425)
)))
