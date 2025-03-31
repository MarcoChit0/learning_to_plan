(define (problem prob)
 (:domain barman)
 (:objects 
      shaker45 - shaker
      left right - hand
      shot497 shot441 shot243 - shot
      ingredient476 ingredient275 ingredient215 - ingredient
      cocktail1 - cocktail
      dispenser450 dispenser181 dispenser34 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker45)
  (ontable shot497)
  (ontable shot441)
  (ontable shot243)
  (dispenses dispenser450 ingredient476)
  (dispenses dispenser181 ingredient275)
  (dispenses dispenser34 ingredient215)
  (clean shaker45)
  (clean shot497)
  (clean shot441)
  (clean shot243)
  (empty shaker45)
  (empty shot497)
  (empty shot441)
  (empty shot243)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker45 l0)
  (shaker-level shaker45 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient476)
  (cocktail-part2 cocktail1 ingredient275)
)
 (:goal
  (and
      (contains shot497 cocktail1)
      (contains shot441 cocktail1)
)))
