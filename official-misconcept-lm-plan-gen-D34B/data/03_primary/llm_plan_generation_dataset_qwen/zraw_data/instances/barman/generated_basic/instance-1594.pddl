(define (problem prob)
 (:domain barman)
 (:objects 
      shaker156 - shaker
      left right - hand
      shot456 shot396 - shot
      ingredient13 ingredient445 - ingredient
      cocktail432 - cocktail
      dispenser34 dispenser403 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker156)
  (ontable shot456)
  (ontable shot396)
  (dispenses dispenser34 ingredient13)
  (dispenses dispenser403 ingredient445)
  (clean shaker156)
  (clean shot456)
  (clean shot396)
  (empty shaker156)
  (empty shot456)
  (empty shot396)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker156 l0)
  (shaker-level shaker156 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail432 ingredient13)
  (cocktail-part2 cocktail432 ingredient445)
)
 (:goal
  (and
      (contains shot456 cocktail432)
)))
