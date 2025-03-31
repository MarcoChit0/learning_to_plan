(define (problem prob)
 (:domain barman)
 (:objects 
      shaker382 - shaker
      left right - hand
      shot415 shot16 shot125 - shot
      ingredient350 ingredient52 ingredient319 ingredient102 - ingredient
      cocktail241 - cocktail
      dispenser475 dispenser482 dispenser280 dispenser376 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker382)
  (ontable shot415)
  (ontable shot16)
  (ontable shot125)
  (dispenses dispenser475 ingredient350)
  (dispenses dispenser482 ingredient52)
  (dispenses dispenser280 ingredient319)
  (dispenses dispenser376 ingredient102)
  (clean shaker382)
  (clean shot415)
  (clean shot16)
  (clean shot125)
  (empty shaker382)
  (empty shot415)
  (empty shot16)
  (empty shot125)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker382 l0)
  (shaker-level shaker382 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail241 ingredient350)
  (cocktail-part2 cocktail241 ingredient52)
)
 (:goal
  (and
      (contains shot415 cocktail241)
      (contains shot16 ingredient350)
)))
