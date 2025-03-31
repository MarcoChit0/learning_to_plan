(define (problem prob)
 (:domain barman)
 (:objects 
      shaker72 - shaker
      left right - hand
      shot50 shot325 - shot
      ingredient331 ingredient14 ingredient89 ingredient142 - ingredient
      cocktail42 - cocktail
      dispenser239 dispenser308 dispenser94 dispenser283 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker72)
  (ontable shot50)
  (ontable shot325)
  (dispenses dispenser239 ingredient331)
  (dispenses dispenser308 ingredient14)
  (dispenses dispenser94 ingredient89)
  (dispenses dispenser283 ingredient142)
  (clean shaker72)
  (clean shot50)
  (clean shot325)
  (empty shaker72)
  (empty shot50)
  (empty shot325)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker72 l0)
  (shaker-level shaker72 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail42 ingredient89)
  (cocktail-part2 cocktail42 ingredient14)
)
 (:goal
  (and
      (contains shot50 cocktail42)
)))
