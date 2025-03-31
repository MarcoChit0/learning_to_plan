(define (problem prob)
 (:domain barman)
 (:objects 
      shaker106 - shaker
      left right - hand
      shot108 - shot
      ingredient157 ingredient208 - ingredient
      cocktail278 - cocktail
      dispenser167 dispenser342 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker106)
  (ontable shot108)
  (dispenses dispenser167 ingredient157)
  (dispenses dispenser342 ingredient208)
  (clean shaker106)
  (clean shot108)
  (empty shaker106)
  (empty shot108)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker106 l0)
  (shaker-level shaker106 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail278 ingredient157)
  (cocktail-part2 cocktail278 ingredient208)
)
 (:goal
  (and
      (contains shot108 cocktail278)
)))
