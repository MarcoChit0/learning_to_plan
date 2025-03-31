(define (problem prob)
 (:domain barman)
 (:objects 
      shaker280 - shaker
      left right - hand
      shot445 shot385 - shot
      ingredient226 ingredient286 - ingredient
      cocktail360 - cocktail
      dispenser166 dispenser486 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker280)
  (ontable shot445)
  (ontable shot385)
  (dispenses dispenser166 ingredient226)
  (dispenses dispenser486 ingredient286)
  (clean shaker280)
  (clean shot445)
  (clean shot385)
  (empty shaker280)
  (empty shot445)
  (empty shot385)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker280 l0)
  (shaker-level shaker280 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail360 ingredient226)
  (cocktail-part2 cocktail360 ingredient286)
)
 (:goal
  (and
      (contains shot445 cocktail360)
)))
