(define (problem prob)
 (:domain barman)
 (:objects 
      shaker261 - shaker
      left right - hand
      shot420 shot358 - shot
      ingredient406 ingredient123 - ingredient
      cocktail1 - cocktail
      dispenser236 dispenser118 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker261)
  (ontable shot420)
  (ontable shot358)
  (dispenses dispenser236 ingredient406)
  (dispenses dispenser118 ingredient123)
  (clean shaker261)
  (clean shot420)
  (clean shot358)
  (empty shaker261)
  (empty shot420)
  (empty shot358)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker261 l0)
  (shaker-level shaker261 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient406)
  (cocktail-part2 cocktail1 ingredient123)
)
 (:goal
  (and
      (contains shot420 cocktail1)
)))
