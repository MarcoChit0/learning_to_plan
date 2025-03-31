(define (problem prob)
 (:domain barman)
 (:objects 
      shaker20 - shaker
      left right - hand
      shot217 - shot
      ingredient201 ingredient155 ingredient285 - ingredient
      cocktail1 - cocktail
      dispenser393 dispenser436 dispenser79 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker20)
  (ontable shot217)
  (dispenses dispenser393 ingredient201)
  (dispenses dispenser436 ingredient155)
  (dispenses dispenser79 ingredient285)
  (clean shaker20)
  (clean shot217)
  (empty shaker20)
  (empty shot217)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker20 l0)
  (shaker-level shaker20 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient285)
  (cocktail-part2 cocktail1 ingredient201)
)
 (:goal
  (and
      (contains shot217 cocktail1)
)))
