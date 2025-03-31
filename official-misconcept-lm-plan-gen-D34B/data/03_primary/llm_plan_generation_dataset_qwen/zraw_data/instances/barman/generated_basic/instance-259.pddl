(define (problem prob)
 (:domain barman)
 (:objects 
      shaker308 - shaker
      left right - hand
      shot466 shot473 shot462 - shot
      ingredient451 ingredient495 ingredient363 - ingredient
      cocktail1 - cocktail
      dispenser453 dispenser394 dispenser294 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker308)
  (ontable shot466)
  (ontable shot473)
  (ontable shot462)
  (dispenses dispenser453 ingredient451)
  (dispenses dispenser394 ingredient495)
  (dispenses dispenser294 ingredient363)
  (clean shaker308)
  (clean shot466)
  (clean shot473)
  (clean shot462)
  (empty shaker308)
  (empty shot466)
  (empty shot473)
  (empty shot462)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker308 l0)
  (shaker-level shaker308 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient495)
  (cocktail-part2 cocktail1 ingredient451)
)
 (:goal
  (and
      (contains shot466 cocktail1)
      (contains shot473 cocktail1)
)))
