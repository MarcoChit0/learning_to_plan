(define (problem prob)
 (:domain barman)
 (:objects 
      shaker497 - shaker
      left right - hand
      shot266 shot369 - shot
      ingredient460 ingredient441 - ingredient
      cocktail1 - cocktail
      dispenser444 dispenser44 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker497)
  (ontable shot266)
  (ontable shot369)
  (dispenses dispenser444 ingredient460)
  (dispenses dispenser44 ingredient441)
  (clean shaker497)
  (clean shot266)
  (clean shot369)
  (empty shaker497)
  (empty shot266)
  (empty shot369)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker497 l0)
  (shaker-level shaker497 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient441)
  (cocktail-part2 cocktail1 ingredient460)
)
 (:goal
  (and
      (contains shot266 cocktail1)
)))
