(define (problem prob)
 (:domain barman)
 (:objects 
      shaker424 - shaker
      left right - hand
      shot430 shot269 - shot
      ingredient499 ingredient98 - ingredient
      cocktail252 - cocktail
      dispenser311 dispenser28 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker424)
  (ontable shot430)
  (ontable shot269)
  (dispenses dispenser311 ingredient499)
  (dispenses dispenser28 ingredient98)
  (clean shaker424)
  (clean shot430)
  (clean shot269)
  (empty shaker424)
  (empty shot430)
  (empty shot269)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker424 l0)
  (shaker-level shaker424 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail252 ingredient499)
  (cocktail-part2 cocktail252 ingredient98)
)
 (:goal
  (and
      (contains shot430 cocktail252)
)))
