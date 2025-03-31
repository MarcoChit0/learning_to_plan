(define (problem prob)
 (:domain barman)
 (:objects 
      shaker327 - shaker
      left right - hand
      shot34 - shot
      ingredient380 ingredient168 ingredient315 ingredient294 - ingredient
      cocktail443 - cocktail
      dispenser204 dispenser196 dispenser3 dispenser369 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker327)
  (ontable shot34)
  (dispenses dispenser204 ingredient380)
  (dispenses dispenser196 ingredient168)
  (dispenses dispenser3 ingredient315)
  (dispenses dispenser369 ingredient294)
  (clean shaker327)
  (clean shot34)
  (empty shaker327)
  (empty shot34)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker327 l0)
  (shaker-level shaker327 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail443 ingredient380)
  (cocktail-part2 cocktail443 ingredient168)
)
 (:goal
  (and
      (contains shot34 cocktail443)
)))
