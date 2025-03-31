(define (problem prob)
 (:domain barman)
 (:objects 
      shaker98 - shaker
      left right - hand
      shot140 - shot
      ingredient73 ingredient367 ingredient254 - ingredient
      cocktail175 - cocktail
      dispenser439 dispenser210 dispenser308 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker98)
  (ontable shot140)
  (dispenses dispenser439 ingredient73)
  (dispenses dispenser210 ingredient367)
  (dispenses dispenser308 ingredient254)
  (clean shaker98)
  (clean shot140)
  (empty shaker98)
  (empty shot140)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker98 l0)
  (shaker-level shaker98 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail175 ingredient254)
  (cocktail-part2 cocktail175 ingredient73)
)
 (:goal
  (and
      (contains shot140 cocktail175)
)))
