(define (problem prob)
 (:domain barman)
 (:objects 
      shaker5 - shaker
      left right - hand
      shot430 - shot
      ingredient281 ingredient51 - ingredient
      cocktail479 - cocktail
      dispenser410 dispenser224 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker5)
  (ontable shot430)
  (dispenses dispenser410 ingredient281)
  (dispenses dispenser224 ingredient51)
  (clean shaker5)
  (clean shot430)
  (empty shaker5)
  (empty shot430)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker5 l0)
  (shaker-level shaker5 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail479 ingredient51)
  (cocktail-part2 cocktail479 ingredient281)
)
 (:goal
  (and
      (contains shot430 cocktail479)
)))
