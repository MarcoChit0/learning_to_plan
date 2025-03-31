(define (problem prob)
 (:domain barman)
 (:objects 
      shaker284 - shaker
      left right - hand
      shot454 shot102 shot149 - shot
      ingredient167 ingredient249 - ingredient
      cocktail70 - cocktail
      dispenser67 dispenser434 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker284)
  (ontable shot454)
  (ontable shot102)
  (ontable shot149)
  (dispenses dispenser67 ingredient167)
  (dispenses dispenser434 ingredient249)
  (clean shaker284)
  (clean shot454)
  (clean shot102)
  (clean shot149)
  (empty shaker284)
  (empty shot454)
  (empty shot102)
  (empty shot149)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker284 l0)
  (shaker-level shaker284 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail70 ingredient167)
  (cocktail-part2 cocktail70 ingredient249)
)
 (:goal
  (and
      (contains shot454 cocktail70)
      (contains shot102 ingredient249)
)))
