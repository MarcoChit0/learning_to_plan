(define (problem prob)
 (:domain barman)
 (:objects 
      shaker233 - shaker
      left right - hand
      shot178 shot182 - shot
      ingredient332 ingredient468 - ingredient
      cocktail113 - cocktail
      dispenser118 dispenser132 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker233)
  (ontable shot178)
  (ontable shot182)
  (dispenses dispenser118 ingredient332)
  (dispenses dispenser132 ingredient468)
  (clean shaker233)
  (clean shot178)
  (clean shot182)
  (empty shaker233)
  (empty shot178)
  (empty shot182)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker233 l0)
  (shaker-level shaker233 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail113 ingredient332)
  (cocktail-part2 cocktail113 ingredient468)
)
 (:goal
  (and
      (contains shot178 cocktail113)
)))
