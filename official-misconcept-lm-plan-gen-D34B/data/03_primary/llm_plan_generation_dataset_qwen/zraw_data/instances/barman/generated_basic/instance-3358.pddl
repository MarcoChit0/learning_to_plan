(define (problem prob)
 (:domain barman)
 (:objects 
      shaker34 - shaker
      left right - hand
      shot441 - shot
      ingredient445 ingredient203 - ingredient
      cocktail167 - cocktail
      dispenser147 dispenser286 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker34)
  (ontable shot441)
  (dispenses dispenser147 ingredient445)
  (dispenses dispenser286 ingredient203)
  (clean shaker34)
  (clean shot441)
  (empty shaker34)
  (empty shot441)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker34 l0)
  (shaker-level shaker34 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail167 ingredient203)
  (cocktail-part2 cocktail167 ingredient445)
)
 (:goal
  (and
      (contains shot441 cocktail167)
)))
