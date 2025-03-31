(define (problem prob)
 (:domain barman)
 (:objects 
      shaker28 - shaker
      left right - hand
      shot304 shot56 - shot
      ingredient265 ingredient90 ingredient232 - ingredient
      cocktail269 - cocktail
      dispenser87 dispenser452 dispenser83 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker28)
  (ontable shot304)
  (ontable shot56)
  (dispenses dispenser87 ingredient265)
  (dispenses dispenser452 ingredient90)
  (dispenses dispenser83 ingredient232)
  (clean shaker28)
  (clean shot304)
  (clean shot56)
  (empty shaker28)
  (empty shot304)
  (empty shot56)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker28 l0)
  (shaker-level shaker28 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail269 ingredient265)
  (cocktail-part2 cocktail269 ingredient90)
)
 (:goal
  (and
      (contains shot304 cocktail269)
)))
