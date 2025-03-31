(define (problem prob)
 (:domain barman)
 (:objects 
      shaker136 - shaker
      left right - hand
      shot64 - shot
      ingredient463 ingredient400 ingredient146 ingredient262 - ingredient
      cocktail1 - cocktail
      dispenser253 dispenser308 dispenser96 dispenser173 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker136)
  (ontable shot64)
  (dispenses dispenser253 ingredient463)
  (dispenses dispenser308 ingredient400)
  (dispenses dispenser96 ingredient146)
  (dispenses dispenser173 ingredient262)
  (clean shaker136)
  (clean shot64)
  (empty shaker136)
  (empty shot64)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker136 l0)
  (shaker-level shaker136 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient146)
  (cocktail-part2 cocktail1 ingredient262)
)
 (:goal
  (and
      (contains shot64 cocktail1)
)))
