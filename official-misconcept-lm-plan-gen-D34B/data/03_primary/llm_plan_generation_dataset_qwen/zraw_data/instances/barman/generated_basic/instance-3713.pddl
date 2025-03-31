(define (problem prob)
 (:domain barman)
 (:objects 
      shaker248 - shaker
      left right - hand
      shot294 - shot
      ingredient445 ingredient494 ingredient139 ingredient155 - ingredient
      cocktail129 - cocktail
      dispenser88 dispenser176 dispenser34 dispenser436 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker248)
  (ontable shot294)
  (dispenses dispenser88 ingredient445)
  (dispenses dispenser176 ingredient494)
  (dispenses dispenser34 ingredient139)
  (dispenses dispenser436 ingredient155)
  (clean shaker248)
  (clean shot294)
  (empty shaker248)
  (empty shot294)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker248 l0)
  (shaker-level shaker248 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail129 ingredient155)
  (cocktail-part2 cocktail129 ingredient494)
)
 (:goal
  (and
      (contains shot294 cocktail129)
)))
