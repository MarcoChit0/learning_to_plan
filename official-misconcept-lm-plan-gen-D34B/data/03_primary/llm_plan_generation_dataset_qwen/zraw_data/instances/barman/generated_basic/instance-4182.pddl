(define (problem prob)
 (:domain barman)
 (:objects 
      shaker398 - shaker
      left right - hand
      shot150 - shot
      ingredient175 ingredient113 ingredient286 ingredient468 - ingredient
      cocktail95 - cocktail
      dispenser388 dispenser16 dispenser22 dispenser176 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker398)
  (ontable shot150)
  (dispenses dispenser388 ingredient175)
  (dispenses dispenser16 ingredient113)
  (dispenses dispenser22 ingredient286)
  (dispenses dispenser176 ingredient468)
  (clean shaker398)
  (clean shot150)
  (empty shaker398)
  (empty shot150)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker398 l0)
  (shaker-level shaker398 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail95 ingredient175)
  (cocktail-part2 cocktail95 ingredient113)
)
 (:goal
  (and
      (contains shot150 cocktail95)
)))
