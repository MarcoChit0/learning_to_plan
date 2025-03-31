(define (problem prob)
 (:domain barman)
 (:objects 
      shaker445 - shaker
      left right - hand
      shot328 shot277 - shot
      ingredient164 ingredient184 ingredient78 ingredient21 - ingredient
      cocktail81 - cocktail
      dispenser309 dispenser251 dispenser215 dispenser352 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker445)
  (ontable shot328)
  (ontable shot277)
  (dispenses dispenser309 ingredient164)
  (dispenses dispenser251 ingredient184)
  (dispenses dispenser215 ingredient78)
  (dispenses dispenser352 ingredient21)
  (clean shaker445)
  (clean shot328)
  (clean shot277)
  (empty shaker445)
  (empty shot328)
  (empty shot277)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker445 l0)
  (shaker-level shaker445 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail81 ingredient164)
  (cocktail-part2 cocktail81 ingredient78)
)
 (:goal
  (and
      (contains shot328 cocktail81)
)))
