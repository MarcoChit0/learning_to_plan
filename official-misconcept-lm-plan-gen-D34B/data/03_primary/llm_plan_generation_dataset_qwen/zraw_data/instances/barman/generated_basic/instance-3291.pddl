(define (problem prob)
 (:domain barman)
 (:objects 
      shaker365 - shaker
      left right - hand
      shot437 - shot
      ingredient462 ingredient490 - ingredient
      cocktail478 - cocktail
      dispenser240 dispenser479 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker365)
  (ontable shot437)
  (dispenses dispenser240 ingredient462)
  (dispenses dispenser479 ingredient490)
  (clean shaker365)
  (clean shot437)
  (empty shaker365)
  (empty shot437)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker365 l0)
  (shaker-level shaker365 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail478 ingredient462)
  (cocktail-part2 cocktail478 ingredient490)
)
 (:goal
  (and
      (contains shot437 cocktail478)
)))
