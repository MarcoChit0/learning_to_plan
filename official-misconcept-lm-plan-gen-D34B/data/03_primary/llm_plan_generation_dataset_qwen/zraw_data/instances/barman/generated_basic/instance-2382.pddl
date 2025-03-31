(define (problem prob)
 (:domain barman)
 (:objects 
      shaker210 - shaker
      left right - hand
      shot358 - shot
      ingredient435 ingredient284 ingredient108 ingredient18 - ingredient
      cocktail474 - cocktail
      dispenser5 dispenser312 dispenser0 dispenser258 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker210)
  (ontable shot358)
  (dispenses dispenser5 ingredient435)
  (dispenses dispenser312 ingredient284)
  (dispenses dispenser0 ingredient108)
  (dispenses dispenser258 ingredient18)
  (clean shaker210)
  (clean shot358)
  (empty shaker210)
  (empty shot358)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker210 l0)
  (shaker-level shaker210 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail474 ingredient284)
  (cocktail-part2 cocktail474 ingredient435)
)
 (:goal
  (and
      (contains shot358 cocktail474)
)))
