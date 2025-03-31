(define (problem prob)
 (:domain barman)
 (:objects 
      shaker466 - shaker
      left right - hand
      shot105 shot479 shot497 - shot
      ingredient290 ingredient390 ingredient369 - ingredient
      cocktail255 - cocktail
      dispenser333 dispenser294 dispenser456 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker466)
  (ontable shot105)
  (ontable shot479)
  (ontable shot497)
  (dispenses dispenser333 ingredient290)
  (dispenses dispenser294 ingredient390)
  (dispenses dispenser456 ingredient369)
  (clean shaker466)
  (clean shot105)
  (clean shot479)
  (clean shot497)
  (empty shaker466)
  (empty shot105)
  (empty shot479)
  (empty shot497)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker466 l0)
  (shaker-level shaker466 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail255 ingredient369)
  (cocktail-part2 cocktail255 ingredient390)
)
 (:goal
  (and
      (contains shot105 cocktail255)
      (contains shot479 ingredient369)
)))
