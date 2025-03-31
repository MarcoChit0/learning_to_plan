(define (problem prob)
 (:domain barman)
 (:objects 
      shaker231 - shaker
      left right - hand
      shot462 - shot
      ingredient82 ingredient487 ingredient148 - ingredient
      cocktail395 - cocktail
      dispenser480 dispenser195 dispenser474 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker231)
  (ontable shot462)
  (dispenses dispenser480 ingredient82)
  (dispenses dispenser195 ingredient487)
  (dispenses dispenser474 ingredient148)
  (clean shaker231)
  (clean shot462)
  (empty shaker231)
  (empty shot462)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker231 l0)
  (shaker-level shaker231 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail395 ingredient148)
  (cocktail-part2 cocktail395 ingredient82)
)
 (:goal
  (and
      (contains shot462 cocktail395)
)))
