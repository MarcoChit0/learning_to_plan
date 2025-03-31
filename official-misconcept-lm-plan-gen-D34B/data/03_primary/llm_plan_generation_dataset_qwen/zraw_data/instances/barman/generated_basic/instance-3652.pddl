(define (problem prob)
 (:domain barman)
 (:objects 
      shaker478 - shaker
      left right - hand
      shot148 shot204 - shot
      ingredient417 ingredient350 ingredient116 ingredient135 - ingredient
      cocktail83 - cocktail
      dispenser305 dispenser421 dispenser38 dispenser387 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker478)
  (ontable shot148)
  (ontable shot204)
  (dispenses dispenser305 ingredient417)
  (dispenses dispenser421 ingredient350)
  (dispenses dispenser38 ingredient116)
  (dispenses dispenser387 ingredient135)
  (clean shaker478)
  (clean shot148)
  (clean shot204)
  (empty shaker478)
  (empty shot148)
  (empty shot204)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker478 l0)
  (shaker-level shaker478 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail83 ingredient116)
  (cocktail-part2 cocktail83 ingredient350)
)
 (:goal
  (and
      (contains shot148 cocktail83)
)))
