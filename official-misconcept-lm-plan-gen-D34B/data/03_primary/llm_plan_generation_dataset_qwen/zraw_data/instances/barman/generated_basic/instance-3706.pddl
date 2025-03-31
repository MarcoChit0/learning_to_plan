(define (problem prob)
 (:domain barman)
 (:objects 
      shaker236 - shaker
      left right - hand
      shot381 shot200 - shot
      ingredient56 ingredient13 ingredient83 - ingredient
      cocktail489 - cocktail
      dispenser440 dispenser158 dispenser219 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker236)
  (ontable shot381)
  (ontable shot200)
  (dispenses dispenser440 ingredient56)
  (dispenses dispenser158 ingredient13)
  (dispenses dispenser219 ingredient83)
  (clean shaker236)
  (clean shot381)
  (clean shot200)
  (empty shaker236)
  (empty shot381)
  (empty shot200)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker236 l0)
  (shaker-level shaker236 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail489 ingredient13)
  (cocktail-part2 cocktail489 ingredient83)
)
 (:goal
  (and
      (contains shot381 cocktail489)
)))
