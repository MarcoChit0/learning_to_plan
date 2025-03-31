(define (problem prob)
 (:domain barman)
 (:objects 
      shaker293 - shaker
      left right - hand
      shot315 shot471 - shot
      ingredient136 ingredient174 ingredient377 - ingredient
      cocktail424 - cocktail
      dispenser253 dispenser232 dispenser31 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker293)
  (ontable shot315)
  (ontable shot471)
  (dispenses dispenser253 ingredient136)
  (dispenses dispenser232 ingredient174)
  (dispenses dispenser31 ingredient377)
  (clean shaker293)
  (clean shot315)
  (clean shot471)
  (empty shaker293)
  (empty shot315)
  (empty shot471)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker293 l0)
  (shaker-level shaker293 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail424 ingredient377)
  (cocktail-part2 cocktail424 ingredient174)
)
 (:goal
  (and
      (contains shot315 cocktail424)
)))
