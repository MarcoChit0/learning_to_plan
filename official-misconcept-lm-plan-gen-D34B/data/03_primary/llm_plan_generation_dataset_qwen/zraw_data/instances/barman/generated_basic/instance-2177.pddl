(define (problem prob)
 (:domain barman)
 (:objects 
      shaker380 - shaker
      left right - hand
      shot131 shot177 - shot
      ingredient157 ingredient362 - ingredient
      cocktail457 - cocktail
      dispenser339 dispenser320 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker380)
  (ontable shot131)
  (ontable shot177)
  (dispenses dispenser339 ingredient157)
  (dispenses dispenser320 ingredient362)
  (clean shaker380)
  (clean shot131)
  (clean shot177)
  (empty shaker380)
  (empty shot131)
  (empty shot177)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker380 l0)
  (shaker-level shaker380 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail457 ingredient157)
  (cocktail-part2 cocktail457 ingredient362)
)
 (:goal
  (and
      (contains shot131 cocktail457)
)))
