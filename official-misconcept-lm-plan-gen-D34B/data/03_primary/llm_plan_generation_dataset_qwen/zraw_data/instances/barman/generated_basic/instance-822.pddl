(define (problem prob)
 (:domain barman)
 (:objects 
      shaker119 - shaker
      left right - hand
      shot228 shot235 - shot
      ingredient152 ingredient27 ingredient12 ingredient5 - ingredient
      cocktail1 - cocktail
      dispenser124 dispenser84 dispenser280 dispenser452 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker119)
  (ontable shot228)
  (ontable shot235)
  (dispenses dispenser124 ingredient152)
  (dispenses dispenser84 ingredient27)
  (dispenses dispenser280 ingredient12)
  (dispenses dispenser452 ingredient5)
  (clean shaker119)
  (clean shot228)
  (clean shot235)
  (empty shaker119)
  (empty shot228)
  (empty shot235)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker119 l0)
  (shaker-level shaker119 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient12)
  (cocktail-part2 cocktail1 ingredient152)
)
 (:goal
  (and
      (contains shot228 cocktail1)
)))
