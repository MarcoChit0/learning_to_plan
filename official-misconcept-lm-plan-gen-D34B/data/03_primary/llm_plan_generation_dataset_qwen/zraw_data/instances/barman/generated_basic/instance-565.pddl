(define (problem prob)
 (:domain barman)
 (:objects 
      shaker24 - shaker
      left right - hand
      shot125 shot137 - shot
      ingredient453 ingredient72 ingredient476 - ingredient
      cocktail1 - cocktail
      dispenser248 dispenser448 dispenser368 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker24)
  (ontable shot125)
  (ontable shot137)
  (dispenses dispenser248 ingredient453)
  (dispenses dispenser448 ingredient72)
  (dispenses dispenser368 ingredient476)
  (clean shaker24)
  (clean shot125)
  (clean shot137)
  (empty shaker24)
  (empty shot125)
  (empty shot137)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker24 l0)
  (shaker-level shaker24 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient476)
  (cocktail-part2 cocktail1 ingredient453)
)
 (:goal
  (and
      (contains shot125 cocktail1)
)))
