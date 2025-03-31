(define (problem prob)
 (:domain barman)
 (:objects 
      shaker128 - shaker
      left right - hand
      shot94 - shot
      ingredient179 ingredient496 ingredient327 - ingredient
      cocktail136 - cocktail
      dispenser154 dispenser449 dispenser274 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker128)
  (ontable shot94)
  (dispenses dispenser154 ingredient179)
  (dispenses dispenser449 ingredient496)
  (dispenses dispenser274 ingredient327)
  (clean shaker128)
  (clean shot94)
  (empty shaker128)
  (empty shot94)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker128 l0)
  (shaker-level shaker128 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail136 ingredient327)
  (cocktail-part2 cocktail136 ingredient496)
)
 (:goal
  (and
      (contains shot94 cocktail136)
)))
