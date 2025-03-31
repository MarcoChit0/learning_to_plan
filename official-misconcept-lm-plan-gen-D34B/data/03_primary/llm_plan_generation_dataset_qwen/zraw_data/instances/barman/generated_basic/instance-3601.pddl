(define (problem prob)
 (:domain barman)
 (:objects 
      shaker282 - shaker
      left right - hand
      shot320 shot433 shot94 shot100 - shot
      ingredient416 ingredient227 ingredient445 - ingredient
      cocktail457 - cocktail
      dispenser382 dispenser262 dispenser276 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker282)
  (ontable shot320)
  (ontable shot433)
  (ontable shot94)
  (ontable shot100)
  (dispenses dispenser382 ingredient416)
  (dispenses dispenser262 ingredient227)
  (dispenses dispenser276 ingredient445)
  (clean shaker282)
  (clean shot320)
  (clean shot433)
  (clean shot94)
  (clean shot100)
  (empty shaker282)
  (empty shot320)
  (empty shot433)
  (empty shot94)
  (empty shot100)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker282 l0)
  (shaker-level shaker282 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail457 ingredient445)
  (cocktail-part2 cocktail457 ingredient227)
)
 (:goal
  (and
      (contains shot320 cocktail457)
      (contains shot433 ingredient445)
      (contains shot94 cocktail457)
)))
