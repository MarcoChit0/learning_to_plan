(define (problem prob)
 (:domain barman)
 (:objects 
      shaker139 - shaker
      left right - hand
      shot52 - shot
      ingredient152 ingredient311 - ingredient
      cocktail1 - cocktail
      dispenser448 dispenser44 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker139)
  (ontable shot52)
  (dispenses dispenser448 ingredient152)
  (dispenses dispenser44 ingredient311)
  (clean shaker139)
  (clean shot52)
  (empty shaker139)
  (empty shot52)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker139 l0)
  (shaker-level shaker139 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient152)
  (cocktail-part2 cocktail1 ingredient311)
)
 (:goal
  (and
      (contains shot52 cocktail1)
)))
