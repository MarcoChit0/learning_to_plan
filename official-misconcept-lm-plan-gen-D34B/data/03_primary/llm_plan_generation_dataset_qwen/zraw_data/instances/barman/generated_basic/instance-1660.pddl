(define (problem prob)
 (:domain barman)
 (:objects 
      shaker184 - shaker
      left right - hand
      shot197 shot154 - shot
      ingredient150 ingredient454 ingredient234 ingredient404 - ingredient
      cocktail203 - cocktail
      dispenser199 dispenser458 dispenser256 dispenser323 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker184)
  (ontable shot197)
  (ontable shot154)
  (dispenses dispenser199 ingredient150)
  (dispenses dispenser458 ingredient454)
  (dispenses dispenser256 ingredient234)
  (dispenses dispenser323 ingredient404)
  (clean shaker184)
  (clean shot197)
  (clean shot154)
  (empty shaker184)
  (empty shot197)
  (empty shot154)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker184 l0)
  (shaker-level shaker184 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail203 ingredient234)
  (cocktail-part2 cocktail203 ingredient404)
)
 (:goal
  (and
      (contains shot197 cocktail203)
)))
