(define (problem prob)
 (:domain barman)
 (:objects 
      shaker445 - shaker
      left right - hand
      shot239 shot402 - shot
      ingredient433 ingredient454 ingredient160 - ingredient
      cocktail43 - cocktail
      dispenser127 dispenser441 dispenser336 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker445)
  (ontable shot239)
  (ontable shot402)
  (dispenses dispenser127 ingredient433)
  (dispenses dispenser441 ingredient454)
  (dispenses dispenser336 ingredient160)
  (clean shaker445)
  (clean shot239)
  (clean shot402)
  (empty shaker445)
  (empty shot239)
  (empty shot402)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker445 l0)
  (shaker-level shaker445 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail43 ingredient454)
  (cocktail-part2 cocktail43 ingredient160)
)
 (:goal
  (and
      (contains shot239 cocktail43)
)))
