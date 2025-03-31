(define (problem prob)
 (:domain barman)
 (:objects 
      shaker85 - shaker
      left right - hand
      shot354 shot369 shot142 - shot
      ingredient273 ingredient371 ingredient342 ingredient182 - ingredient
      cocktail127 - cocktail
      dispenser179 dispenser145 dispenser373 dispenser282 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker85)
  (ontable shot354)
  (ontable shot369)
  (ontable shot142)
  (dispenses dispenser179 ingredient273)
  (dispenses dispenser145 ingredient371)
  (dispenses dispenser373 ingredient342)
  (dispenses dispenser282 ingredient182)
  (clean shaker85)
  (clean shot354)
  (clean shot369)
  (clean shot142)
  (empty shaker85)
  (empty shot354)
  (empty shot369)
  (empty shot142)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker85 l0)
  (shaker-level shaker85 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail127 ingredient182)
  (cocktail-part2 cocktail127 ingredient273)
)
 (:goal
  (and
      (contains shot354 cocktail127)
      (contains shot369 cocktail127)
)))
