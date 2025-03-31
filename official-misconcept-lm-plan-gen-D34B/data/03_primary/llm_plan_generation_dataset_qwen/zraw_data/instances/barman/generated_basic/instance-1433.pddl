(define (problem prob)
 (:domain barman)
 (:objects 
      shaker0 - shaker
      left right - hand
      shot180 shot259 - shot
      ingredient190 ingredient281 - ingredient
      cocktail403 - cocktail
      dispenser135 dispenser178 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker0)
  (ontable shot180)
  (ontable shot259)
  (dispenses dispenser135 ingredient190)
  (dispenses dispenser178 ingredient281)
  (clean shaker0)
  (clean shot180)
  (clean shot259)
  (empty shaker0)
  (empty shot180)
  (empty shot259)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker0 l0)
  (shaker-level shaker0 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail403 ingredient281)
  (cocktail-part2 cocktail403 ingredient190)
)
 (:goal
  (and
      (contains shot180 cocktail403)
)))
