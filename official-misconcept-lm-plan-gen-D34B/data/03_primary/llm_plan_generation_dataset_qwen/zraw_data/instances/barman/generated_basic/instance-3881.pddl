(define (problem prob)
 (:domain barman)
 (:objects 
      shaker495 - shaker
      left right - hand
      shot79 shot134 - shot
      ingredient78 ingredient70 ingredient303 - ingredient
      cocktail195 - cocktail
      dispenser428 dispenser271 dispenser429 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker495)
  (ontable shot79)
  (ontable shot134)
  (dispenses dispenser428 ingredient78)
  (dispenses dispenser271 ingredient70)
  (dispenses dispenser429 ingredient303)
  (clean shaker495)
  (clean shot79)
  (clean shot134)
  (empty shaker495)
  (empty shot79)
  (empty shot134)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker495 l0)
  (shaker-level shaker495 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail195 ingredient78)
  (cocktail-part2 cocktail195 ingredient303)
)
 (:goal
  (and
      (contains shot79 cocktail195)
)))
