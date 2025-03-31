(define (problem prob)
 (:domain barman)
 (:objects 
      shaker112 - shaker
      left right - hand
      shot140 - shot
      ingredient440 ingredient310 - ingredient
      cocktail422 - cocktail
      dispenser5 dispenser65 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker112)
  (ontable shot140)
  (dispenses dispenser5 ingredient440)
  (dispenses dispenser65 ingredient310)
  (clean shaker112)
  (clean shot140)
  (empty shaker112)
  (empty shot140)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker112 l0)
  (shaker-level shaker112 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail422 ingredient440)
  (cocktail-part2 cocktail422 ingredient310)
)
 (:goal
  (and
      (contains shot140 cocktail422)
)))
