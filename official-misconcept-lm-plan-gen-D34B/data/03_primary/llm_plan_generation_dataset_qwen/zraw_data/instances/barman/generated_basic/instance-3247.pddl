(define (problem prob)
 (:domain barman)
 (:objects 
      shaker461 - shaker
      left right - hand
      shot488 - shot
      ingredient29 ingredient270 ingredient127 - ingredient
      cocktail497 - cocktail
      dispenser213 dispenser473 dispenser420 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker461)
  (ontable shot488)
  (dispenses dispenser213 ingredient29)
  (dispenses dispenser473 ingredient270)
  (dispenses dispenser420 ingredient127)
  (clean shaker461)
  (clean shot488)
  (empty shaker461)
  (empty shot488)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker461 l0)
  (shaker-level shaker461 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail497 ingredient127)
  (cocktail-part2 cocktail497 ingredient270)
)
 (:goal
  (and
      (contains shot488 cocktail497)
)))
