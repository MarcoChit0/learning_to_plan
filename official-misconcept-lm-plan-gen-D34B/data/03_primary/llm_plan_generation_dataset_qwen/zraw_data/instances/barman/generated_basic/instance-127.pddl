(define (problem prob)
 (:domain barman)
 (:objects 
      shaker33 - shaker
      left right - hand
      shot290 shot333 - shot
      ingredient120 ingredient381 - ingredient
      cocktail1 - cocktail
      dispenser343 dispenser232 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker33)
  (ontable shot290)
  (ontable shot333)
  (dispenses dispenser343 ingredient120)
  (dispenses dispenser232 ingredient381)
  (clean shaker33)
  (clean shot290)
  (clean shot333)
  (empty shaker33)
  (empty shot290)
  (empty shot333)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker33 l0)
  (shaker-level shaker33 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient381)
  (cocktail-part2 cocktail1 ingredient120)
)
 (:goal
  (and
      (contains shot290 cocktail1)
)))
