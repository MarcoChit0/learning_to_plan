(define (problem prob)
 (:domain barman)
 (:objects 
      shaker445 - shaker
      left right - hand
      shot331 - shot
      ingredient294 ingredient136 ingredient169 ingredient361 - ingredient
      cocktail197 - cocktail
      dispenser247 dispenser8 dispenser56 dispenser397 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker445)
  (ontable shot331)
  (dispenses dispenser247 ingredient294)
  (dispenses dispenser8 ingredient136)
  (dispenses dispenser56 ingredient169)
  (dispenses dispenser397 ingredient361)
  (clean shaker445)
  (clean shot331)
  (empty shaker445)
  (empty shot331)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker445 l0)
  (shaker-level shaker445 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail197 ingredient361)
  (cocktail-part2 cocktail197 ingredient169)
)
 (:goal
  (and
      (contains shot331 cocktail197)
)))
