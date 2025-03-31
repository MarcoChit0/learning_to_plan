(define (problem prob)
 (:domain barman)
 (:objects 
      shaker32 - shaker
      left right - hand
      shot180 - shot
      ingredient296 ingredient197 - ingredient
      cocktail1 - cocktail
      dispenser112 dispenser416 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker32)
  (ontable shot180)
  (dispenses dispenser112 ingredient296)
  (dispenses dispenser416 ingredient197)
  (clean shaker32)
  (clean shot180)
  (empty shaker32)
  (empty shot180)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker32 l0)
  (shaker-level shaker32 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient296)
  (cocktail-part2 cocktail1 ingredient197)
)
 (:goal
  (and
      (contains shot180 cocktail1)
)))
