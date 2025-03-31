(define (problem prob)
 (:domain barman)
 (:objects 
      shaker284 - shaker
      left right - hand
      shot108 shot275 - shot
      ingredient348 ingredient445 ingredient497 ingredient250 - ingredient
      cocktail431 - cocktail
      dispenser262 dispenser212 dispenser97 dispenser430 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker284)
  (ontable shot108)
  (ontable shot275)
  (dispenses dispenser262 ingredient348)
  (dispenses dispenser212 ingredient445)
  (dispenses dispenser97 ingredient497)
  (dispenses dispenser430 ingredient250)
  (clean shaker284)
  (clean shot108)
  (clean shot275)
  (empty shaker284)
  (empty shot108)
  (empty shot275)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker284 l0)
  (shaker-level shaker284 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail431 ingredient250)
  (cocktail-part2 cocktail431 ingredient445)
)
 (:goal
  (and
      (contains shot108 cocktail431)
)))
