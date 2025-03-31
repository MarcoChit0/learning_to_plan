(define (problem prob)
 (:domain barman)
 (:objects 
      shaker363 - shaker
      left right - hand
      shot229 shot161 - shot
      ingredient324 ingredient461 - ingredient
      cocktail24 - cocktail
      dispenser432 dispenser45 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker363)
  (ontable shot229)
  (ontable shot161)
  (dispenses dispenser432 ingredient324)
  (dispenses dispenser45 ingredient461)
  (clean shaker363)
  (clean shot229)
  (clean shot161)
  (empty shaker363)
  (empty shot229)
  (empty shot161)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker363 l0)
  (shaker-level shaker363 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail24 ingredient324)
  (cocktail-part2 cocktail24 ingredient461)
)
 (:goal
  (and
      (contains shot229 cocktail24)
)))
