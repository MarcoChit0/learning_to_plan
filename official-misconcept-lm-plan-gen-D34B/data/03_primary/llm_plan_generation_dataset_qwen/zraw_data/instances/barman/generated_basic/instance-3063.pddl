(define (problem prob)
 (:domain barman)
 (:objects 
      shaker315 - shaker
      left right - hand
      shot88 shot303 - shot
      ingredient386 ingredient0 - ingredient
      cocktail360 - cocktail
      dispenser68 dispenser332 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker315)
  (ontable shot88)
  (ontable shot303)
  (dispenses dispenser68 ingredient386)
  (dispenses dispenser332 ingredient0)
  (clean shaker315)
  (clean shot88)
  (clean shot303)
  (empty shaker315)
  (empty shot88)
  (empty shot303)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker315 l0)
  (shaker-level shaker315 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail360 ingredient386)
  (cocktail-part2 cocktail360 ingredient0)
)
 (:goal
  (and
      (contains shot88 cocktail360)
)))
