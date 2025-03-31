(define (problem prob)
 (:domain barman)
 (:objects 
      shaker361 - shaker
      left right - hand
      shot208 shot340 shot355 - shot
      ingredient471 ingredient387 ingredient135 - ingredient
      cocktail304 - cocktail
      dispenser272 dispenser255 dispenser19 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker361)
  (ontable shot208)
  (ontable shot340)
  (ontable shot355)
  (dispenses dispenser272 ingredient471)
  (dispenses dispenser255 ingredient387)
  (dispenses dispenser19 ingredient135)
  (clean shaker361)
  (clean shot208)
  (clean shot340)
  (clean shot355)
  (empty shaker361)
  (empty shot208)
  (empty shot340)
  (empty shot355)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker361 l0)
  (shaker-level shaker361 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail304 ingredient471)
  (cocktail-part2 cocktail304 ingredient387)
)
 (:goal
  (and
      (contains shot208 cocktail304)
      (contains shot340 cocktail304)
)))
