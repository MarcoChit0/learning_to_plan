(define (problem prob)
 (:domain barman)
 (:objects 
      shaker97 - shaker
      left right - hand
      shot491 - shot
      ingredient166 ingredient89 ingredient120 ingredient52 - ingredient
      cocktail224 - cocktail
      dispenser48 dispenser264 dispenser115 dispenser447 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker97)
  (ontable shot491)
  (dispenses dispenser48 ingredient166)
  (dispenses dispenser264 ingredient89)
  (dispenses dispenser115 ingredient120)
  (dispenses dispenser447 ingredient52)
  (clean shaker97)
  (clean shot491)
  (empty shaker97)
  (empty shot491)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker97 l0)
  (shaker-level shaker97 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail224 ingredient166)
  (cocktail-part2 cocktail224 ingredient120)
)
 (:goal
  (and
      (contains shot491 cocktail224)
)))
