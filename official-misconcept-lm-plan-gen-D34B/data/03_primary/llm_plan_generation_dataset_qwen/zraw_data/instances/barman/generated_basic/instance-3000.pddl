(define (problem prob)
 (:domain barman)
 (:objects 
      shaker118 - shaker
      left right - hand
      shot100 shot490 - shot
      ingredient347 ingredient478 ingredient83 - ingredient
      cocktail459 - cocktail
      dispenser441 dispenser320 dispenser176 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker118)
  (ontable shot100)
  (ontable shot490)
  (dispenses dispenser441 ingredient347)
  (dispenses dispenser320 ingredient478)
  (dispenses dispenser176 ingredient83)
  (clean shaker118)
  (clean shot100)
  (clean shot490)
  (empty shaker118)
  (empty shot100)
  (empty shot490)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker118 l0)
  (shaker-level shaker118 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail459 ingredient83)
  (cocktail-part2 cocktail459 ingredient478)
)
 (:goal
  (and
      (contains shot100 cocktail459)
)))
