(define (problem prob)
 (:domain barman)
 (:objects 
      shaker96 - shaker
      left right - hand
      shot207 shot255 shot313 - shot
      ingredient231 ingredient404 - ingredient
      cocktail392 - cocktail
      dispenser378 dispenser226 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker96)
  (ontable shot207)
  (ontable shot255)
  (ontable shot313)
  (dispenses dispenser378 ingredient231)
  (dispenses dispenser226 ingredient404)
  (clean shaker96)
  (clean shot207)
  (clean shot255)
  (clean shot313)
  (empty shaker96)
  (empty shot207)
  (empty shot255)
  (empty shot313)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker96 l0)
  (shaker-level shaker96 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail392 ingredient231)
  (cocktail-part2 cocktail392 ingredient404)
)
 (:goal
  (and
      (contains shot207 cocktail392)
      (contains shot255 ingredient404)
)))
