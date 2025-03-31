(define (problem prob)
 (:domain barman)
 (:objects 
      shaker390 - shaker
      left right - hand
      shot175 shot410 shot499 - shot
      ingredient33 ingredient336 ingredient407 - ingredient
      cocktail412 - cocktail
      dispenser305 dispenser388 dispenser150 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker390)
  (ontable shot175)
  (ontable shot410)
  (ontable shot499)
  (dispenses dispenser305 ingredient33)
  (dispenses dispenser388 ingredient336)
  (dispenses dispenser150 ingredient407)
  (clean shaker390)
  (clean shot175)
  (clean shot410)
  (clean shot499)
  (empty shaker390)
  (empty shot175)
  (empty shot410)
  (empty shot499)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker390 l0)
  (shaker-level shaker390 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail412 ingredient33)
  (cocktail-part2 cocktail412 ingredient407)
)
 (:goal
  (and
      (contains shot175 cocktail412)
      (contains shot410 ingredient407)
)))
