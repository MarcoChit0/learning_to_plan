(define (problem prob)
 (:domain barman)
 (:objects 
      shaker13 - shaker
      left right - hand
      shot189 shot77 - shot
      ingredient245 ingredient493 ingredient117 ingredient401 - ingredient
      cocktail194 - cocktail
      dispenser142 dispenser204 dispenser139 dispenser476 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker13)
  (ontable shot189)
  (ontable shot77)
  (dispenses dispenser142 ingredient245)
  (dispenses dispenser204 ingredient493)
  (dispenses dispenser139 ingredient117)
  (dispenses dispenser476 ingredient401)
  (clean shaker13)
  (clean shot189)
  (clean shot77)
  (empty shaker13)
  (empty shot189)
  (empty shot77)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker13 l0)
  (shaker-level shaker13 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail194 ingredient493)
  (cocktail-part2 cocktail194 ingredient401)
)
 (:goal
  (and
      (contains shot189 cocktail194)
)))
