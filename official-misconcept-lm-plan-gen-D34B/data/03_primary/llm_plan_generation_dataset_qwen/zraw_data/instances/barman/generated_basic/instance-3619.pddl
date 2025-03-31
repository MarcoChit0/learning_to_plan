(define (problem prob)
 (:domain barman)
 (:objects 
      shaker20 - shaker
      left right - hand
      shot206 shot102 shot408 - shot
      ingredient298 ingredient109 ingredient283 - ingredient
      cocktail63 - cocktail
      dispenser425 dispenser66 dispenser237 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker20)
  (ontable shot206)
  (ontable shot102)
  (ontable shot408)
  (dispenses dispenser425 ingredient298)
  (dispenses dispenser66 ingredient109)
  (dispenses dispenser237 ingredient283)
  (clean shaker20)
  (clean shot206)
  (clean shot102)
  (clean shot408)
  (empty shaker20)
  (empty shot206)
  (empty shot102)
  (empty shot408)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker20 l0)
  (shaker-level shaker20 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail63 ingredient298)
  (cocktail-part2 cocktail63 ingredient109)
)
 (:goal
  (and
      (contains shot206 cocktail63)
      (contains shot102 cocktail63)
)))
