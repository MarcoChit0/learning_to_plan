(define (problem prob)
 (:domain barman)
 (:objects 
      shaker347 - shaker
      left right - hand
      shot328 shot446 - shot
      ingredient7 ingredient486 ingredient402 ingredient218 - ingredient
      cocktail68 - cocktail
      dispenser460 dispenser344 dispenser169 dispenser482 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker347)
  (ontable shot328)
  (ontable shot446)
  (dispenses dispenser460 ingredient7)
  (dispenses dispenser344 ingredient486)
  (dispenses dispenser169 ingredient402)
  (dispenses dispenser482 ingredient218)
  (clean shaker347)
  (clean shot328)
  (clean shot446)
  (empty shaker347)
  (empty shot328)
  (empty shot446)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker347 l0)
  (shaker-level shaker347 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail68 ingredient7)
  (cocktail-part2 cocktail68 ingredient402)
)
 (:goal
  (and
      (contains shot328 cocktail68)
)))
