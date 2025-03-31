(define (problem prob)
 (:domain barman)
 (:objects 
      shaker0 - shaker
      left right - hand
      shot345 - shot
      ingredient92 ingredient360 ingredient194 - ingredient
      cocktail1 - cocktail
      dispenser498 dispenser56 dispenser199 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker0)
  (ontable shot345)
  (dispenses dispenser498 ingredient92)
  (dispenses dispenser56 ingredient360)
  (dispenses dispenser199 ingredient194)
  (clean shaker0)
  (clean shot345)
  (empty shaker0)
  (empty shot345)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker0 l0)
  (shaker-level shaker0 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient194)
  (cocktail-part2 cocktail1 ingredient92)
)
 (:goal
  (and
      (contains shot345 cocktail1)
)))
