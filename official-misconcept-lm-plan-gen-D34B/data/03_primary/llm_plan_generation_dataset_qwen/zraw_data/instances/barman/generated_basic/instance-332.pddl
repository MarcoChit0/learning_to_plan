(define (problem prob)
 (:domain barman)
 (:objects 
      shaker445 - shaker
      left right - hand
      shot26 - shot
      ingredient330 ingredient107 - ingredient
      cocktail1 - cocktail
      dispenser259 dispenser152 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker445)
  (ontable shot26)
  (dispenses dispenser259 ingredient330)
  (dispenses dispenser152 ingredient107)
  (clean shaker445)
  (clean shot26)
  (empty shaker445)
  (empty shot26)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker445 l0)
  (shaker-level shaker445 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient107)
  (cocktail-part2 cocktail1 ingredient330)
)
 (:goal
  (and
      (contains shot26 cocktail1)
)))
