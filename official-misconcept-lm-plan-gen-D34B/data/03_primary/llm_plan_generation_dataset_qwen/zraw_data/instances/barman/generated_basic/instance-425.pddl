(define (problem prob)
 (:domain barman)
 (:objects 
      shaker376 - shaker
      left right - hand
      shot81 - shot
      ingredient90 ingredient178 - ingredient
      cocktail1 - cocktail
      dispenser366 dispenser117 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker376)
  (ontable shot81)
  (dispenses dispenser366 ingredient90)
  (dispenses dispenser117 ingredient178)
  (clean shaker376)
  (clean shot81)
  (empty shaker376)
  (empty shot81)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker376 l0)
  (shaker-level shaker376 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient178)
  (cocktail-part2 cocktail1 ingredient90)
)
 (:goal
  (and
      (contains shot81 cocktail1)
)))
