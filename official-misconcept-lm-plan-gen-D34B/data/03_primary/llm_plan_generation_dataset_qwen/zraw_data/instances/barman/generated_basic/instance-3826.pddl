(define (problem prob)
 (:domain barman)
 (:objects 
      shaker222 - shaker
      left right - hand
      shot21 shot220 - shot
      ingredient79 ingredient481 ingredient426 - ingredient
      cocktail115 - cocktail
      dispenser403 dispenser275 dispenser171 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker222)
  (ontable shot21)
  (ontable shot220)
  (dispenses dispenser403 ingredient79)
  (dispenses dispenser275 ingredient481)
  (dispenses dispenser171 ingredient426)
  (clean shaker222)
  (clean shot21)
  (clean shot220)
  (empty shaker222)
  (empty shot21)
  (empty shot220)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker222 l0)
  (shaker-level shaker222 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail115 ingredient481)
  (cocktail-part2 cocktail115 ingredient426)
)
 (:goal
  (and
      (contains shot21 cocktail115)
)))
