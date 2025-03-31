(define (problem prob)
 (:domain barman)
 (:objects 
      shaker61 - shaker
      left right - hand
      shot106 - shot
      ingredient364 ingredient377 ingredient40 ingredient318 - ingredient
      cocktail18 - cocktail
      dispenser280 dispenser97 dispenser315 dispenser283 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker61)
  (ontable shot106)
  (dispenses dispenser280 ingredient364)
  (dispenses dispenser97 ingredient377)
  (dispenses dispenser315 ingredient40)
  (dispenses dispenser283 ingredient318)
  (clean shaker61)
  (clean shot106)
  (empty shaker61)
  (empty shot106)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker61 l0)
  (shaker-level shaker61 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail18 ingredient364)
  (cocktail-part2 cocktail18 ingredient377)
)
 (:goal
  (and
      (contains shot106 cocktail18)
)))
