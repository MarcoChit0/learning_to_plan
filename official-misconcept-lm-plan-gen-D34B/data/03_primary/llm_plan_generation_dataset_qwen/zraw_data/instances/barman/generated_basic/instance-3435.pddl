(define (problem prob)
 (:domain barman)
 (:objects 
      shaker56 - shaker
      left right - hand
      shot428 shot302 - shot
      ingredient458 ingredient243 ingredient70 ingredient320 - ingredient
      cocktail308 - cocktail
      dispenser230 dispenser381 dispenser408 dispenser128 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker56)
  (ontable shot428)
  (ontable shot302)
  (dispenses dispenser230 ingredient458)
  (dispenses dispenser381 ingredient243)
  (dispenses dispenser408 ingredient70)
  (dispenses dispenser128 ingredient320)
  (clean shaker56)
  (clean shot428)
  (clean shot302)
  (empty shaker56)
  (empty shot428)
  (empty shot302)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker56 l0)
  (shaker-level shaker56 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail308 ingredient320)
  (cocktail-part2 cocktail308 ingredient243)
)
 (:goal
  (and
      (contains shot428 cocktail308)
)))
