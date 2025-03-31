(define (problem prob)
 (:domain barman)
 (:objects 
      shaker13 - shaker
      left right - hand
      shot339 - shot
      ingredient467 ingredient293 - ingredient
      cocktail1 - cocktail
      dispenser115 dispenser29 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker13)
  (ontable shot339)
  (dispenses dispenser115 ingredient467)
  (dispenses dispenser29 ingredient293)
  (clean shaker13)
  (clean shot339)
  (empty shaker13)
  (empty shot339)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker13 l0)
  (shaker-level shaker13 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient293)
  (cocktail-part2 cocktail1 ingredient467)
)
 (:goal
  (and
      (contains shot339 cocktail1)
)))
