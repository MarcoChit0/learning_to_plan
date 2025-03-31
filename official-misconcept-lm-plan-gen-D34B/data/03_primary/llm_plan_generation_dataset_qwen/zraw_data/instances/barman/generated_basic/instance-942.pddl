(define (problem prob)
 (:domain barman)
 (:objects 
      shaker83 - shaker
      left right - hand
      shot101 shot226 shot84 - shot
      ingredient415 ingredient0 ingredient245 ingredient150 - ingredient
      cocktail1 - cocktail
      dispenser231 dispenser415 dispenser110 dispenser419 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker83)
  (ontable shot101)
  (ontable shot226)
  (ontable shot84)
  (dispenses dispenser231 ingredient415)
  (dispenses dispenser415 ingredient0)
  (dispenses dispenser110 ingredient245)
  (dispenses dispenser419 ingredient150)
  (clean shaker83)
  (clean shot101)
  (clean shot226)
  (clean shot84)
  (empty shaker83)
  (empty shot101)
  (empty shot226)
  (empty shot84)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker83 l0)
  (shaker-level shaker83 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient150)
  (cocktail-part2 cocktail1 ingredient245)
)
 (:goal
  (and
      (contains shot101 cocktail1)
      (contains shot226 cocktail1)
)))
