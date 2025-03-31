(define (problem prob)
 (:domain barman)
 (:objects 
      shaker294 - shaker
      left right - hand
      shot445 - shot
      ingredient340 ingredient151 ingredient420 - ingredient
      cocktail384 - cocktail
      dispenser136 dispenser259 dispenser469 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker294)
  (ontable shot445)
  (dispenses dispenser136 ingredient340)
  (dispenses dispenser259 ingredient151)
  (dispenses dispenser469 ingredient420)
  (clean shaker294)
  (clean shot445)
  (empty shaker294)
  (empty shot445)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker294 l0)
  (shaker-level shaker294 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail384 ingredient420)
  (cocktail-part2 cocktail384 ingredient151)
)
 (:goal
  (and
      (contains shot445 cocktail384)
)))
