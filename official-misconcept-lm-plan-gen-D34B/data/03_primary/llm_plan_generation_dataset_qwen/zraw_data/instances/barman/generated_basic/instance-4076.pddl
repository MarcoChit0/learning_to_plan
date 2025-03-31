(define (problem prob)
 (:domain barman)
 (:objects 
      shaker269 - shaker
      left right - hand
      shot1 shot403 - shot
      ingredient448 ingredient27 ingredient479 ingredient212 - ingredient
      cocktail118 - cocktail
      dispenser9 dispenser396 dispenser101 dispenser118 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker269)
  (ontable shot1)
  (ontable shot403)
  (dispenses dispenser9 ingredient448)
  (dispenses dispenser396 ingredient27)
  (dispenses dispenser101 ingredient479)
  (dispenses dispenser118 ingredient212)
  (clean shaker269)
  (clean shot1)
  (clean shot403)
  (empty shaker269)
  (empty shot1)
  (empty shot403)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker269 l0)
  (shaker-level shaker269 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail118 ingredient27)
  (cocktail-part2 cocktail118 ingredient212)
)
 (:goal
  (and
      (contains shot1 cocktail118)
)))
