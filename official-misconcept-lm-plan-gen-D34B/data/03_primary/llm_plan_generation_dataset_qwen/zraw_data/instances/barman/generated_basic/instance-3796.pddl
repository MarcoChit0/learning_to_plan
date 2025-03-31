(define (problem prob)
 (:domain barman)
 (:objects 
      shaker405 - shaker
      left right - hand
      shot476 - shot
      ingredient410 ingredient77 ingredient475 ingredient251 - ingredient
      cocktail80 - cocktail
      dispenser75 dispenser478 dispenser136 dispenser205 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker405)
  (ontable shot476)
  (dispenses dispenser75 ingredient410)
  (dispenses dispenser478 ingredient77)
  (dispenses dispenser136 ingredient475)
  (dispenses dispenser205 ingredient251)
  (clean shaker405)
  (clean shot476)
  (empty shaker405)
  (empty shot476)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker405 l0)
  (shaker-level shaker405 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail80 ingredient77)
  (cocktail-part2 cocktail80 ingredient251)
)
 (:goal
  (and
      (contains shot476 cocktail80)
)))
