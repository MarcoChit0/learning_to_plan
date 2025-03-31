(define (problem prob)
 (:domain barman)
 (:objects 
      shaker66 - shaker
      left right - hand
      shot51 - shot
      ingredient273 ingredient401 ingredient245 - ingredient
      cocktail114 - cocktail
      dispenser278 dispenser399 dispenser141 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker66)
  (ontable shot51)
  (dispenses dispenser278 ingredient273)
  (dispenses dispenser399 ingredient401)
  (dispenses dispenser141 ingredient245)
  (clean shaker66)
  (clean shot51)
  (empty shaker66)
  (empty shot51)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker66 l0)
  (shaker-level shaker66 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail114 ingredient245)
  (cocktail-part2 cocktail114 ingredient273)
)
 (:goal
  (and
      (contains shot51 cocktail114)
)))
