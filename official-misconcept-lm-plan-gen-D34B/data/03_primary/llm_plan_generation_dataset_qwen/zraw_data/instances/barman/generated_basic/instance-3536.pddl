(define (problem prob)
 (:domain barman)
 (:objects 
      shaker18 - shaker
      left right - hand
      shot448 - shot
      ingredient234 ingredient314 ingredient220 - ingredient
      cocktail369 - cocktail
      dispenser73 dispenser141 dispenser31 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker18)
  (ontable shot448)
  (dispenses dispenser73 ingredient234)
  (dispenses dispenser141 ingredient314)
  (dispenses dispenser31 ingredient220)
  (clean shaker18)
  (clean shot448)
  (empty shaker18)
  (empty shot448)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker18 l0)
  (shaker-level shaker18 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail369 ingredient314)
  (cocktail-part2 cocktail369 ingredient234)
)
 (:goal
  (and
      (contains shot448 cocktail369)
)))
