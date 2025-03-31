(define (problem prob)
 (:domain barman)
 (:objects 
      shaker455 - shaker
      left right - hand
      shot160 shot56 - shot
      ingredient318 ingredient349 ingredient214 - ingredient
      cocktail359 - cocktail
      dispenser373 dispenser69 dispenser378 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker455)
  (ontable shot160)
  (ontable shot56)
  (dispenses dispenser373 ingredient318)
  (dispenses dispenser69 ingredient349)
  (dispenses dispenser378 ingredient214)
  (clean shaker455)
  (clean shot160)
  (clean shot56)
  (empty shaker455)
  (empty shot160)
  (empty shot56)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker455 l0)
  (shaker-level shaker455 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail359 ingredient318)
  (cocktail-part2 cocktail359 ingredient214)
)
 (:goal
  (and
      (contains shot160 cocktail359)
)))
