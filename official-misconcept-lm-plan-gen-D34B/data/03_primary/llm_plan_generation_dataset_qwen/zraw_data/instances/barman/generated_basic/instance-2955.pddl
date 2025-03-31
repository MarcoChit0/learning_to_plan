(define (problem prob)
 (:domain barman)
 (:objects 
      shaker386 - shaker
      left right - hand
      shot137 - shot
      ingredient18 ingredient206 ingredient64 ingredient15 - ingredient
      cocktail97 - cocktail
      dispenser487 dispenser84 dispenser64 dispenser494 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker386)
  (ontable shot137)
  (dispenses dispenser487 ingredient18)
  (dispenses dispenser84 ingredient206)
  (dispenses dispenser64 ingredient64)
  (dispenses dispenser494 ingredient15)
  (clean shaker386)
  (clean shot137)
  (empty shaker386)
  (empty shot137)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker386 l0)
  (shaker-level shaker386 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail97 ingredient18)
  (cocktail-part2 cocktail97 ingredient15)
)
 (:goal
  (and
      (contains shot137 cocktail97)
)))
