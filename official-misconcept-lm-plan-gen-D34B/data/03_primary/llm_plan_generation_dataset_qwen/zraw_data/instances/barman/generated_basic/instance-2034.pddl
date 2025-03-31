(define (problem prob)
 (:domain barman)
 (:objects 
      shaker472 - shaker
      left right - hand
      shot108 shot172 - shot
      ingredient154 ingredient214 - ingredient
      cocktail164 - cocktail
      dispenser342 dispenser23 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker472)
  (ontable shot108)
  (ontable shot172)
  (dispenses dispenser342 ingredient154)
  (dispenses dispenser23 ingredient214)
  (clean shaker472)
  (clean shot108)
  (clean shot172)
  (empty shaker472)
  (empty shot108)
  (empty shot172)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker472 l0)
  (shaker-level shaker472 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail164 ingredient154)
  (cocktail-part2 cocktail164 ingredient214)
)
 (:goal
  (and
      (contains shot108 cocktail164)
)))
