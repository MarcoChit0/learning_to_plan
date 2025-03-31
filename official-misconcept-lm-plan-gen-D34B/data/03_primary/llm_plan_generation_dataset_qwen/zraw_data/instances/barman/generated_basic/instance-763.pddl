(define (problem prob)
 (:domain barman)
 (:objects 
      shaker376 - shaker
      left right - hand
      shot467 - shot
      ingredient393 ingredient305 - ingredient
      cocktail1 - cocktail
      dispenser344 dispenser115 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker376)
  (ontable shot467)
  (dispenses dispenser344 ingredient393)
  (dispenses dispenser115 ingredient305)
  (clean shaker376)
  (clean shot467)
  (empty shaker376)
  (empty shot467)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker376 l0)
  (shaker-level shaker376 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient393)
  (cocktail-part2 cocktail1 ingredient305)
)
 (:goal
  (and
      (contains shot467 cocktail1)
)))
