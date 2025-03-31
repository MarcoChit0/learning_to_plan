(define (problem prob)
 (:domain barman)
 (:objects 
      shaker28 - shaker
      left right - hand
      shot380 shot196 - shot
      ingredient266 ingredient117 - ingredient
      cocktail178 - cocktail
      dispenser86 dispenser46 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker28)
  (ontable shot380)
  (ontable shot196)
  (dispenses dispenser86 ingredient266)
  (dispenses dispenser46 ingredient117)
  (clean shaker28)
  (clean shot380)
  (clean shot196)
  (empty shaker28)
  (empty shot380)
  (empty shot196)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker28 l0)
  (shaker-level shaker28 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail178 ingredient117)
  (cocktail-part2 cocktail178 ingredient266)
)
 (:goal
  (and
      (contains shot380 cocktail178)
)))
