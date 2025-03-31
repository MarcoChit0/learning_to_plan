(define (problem prob)
 (:domain barman)
 (:objects 
      shaker205 - shaker
      left right - hand
      shot296 - shot
      ingredient437 ingredient457 - ingredient
      cocktail5 - cocktail
      dispenser135 dispenser29 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker205)
  (ontable shot296)
  (dispenses dispenser135 ingredient437)
  (dispenses dispenser29 ingredient457)
  (clean shaker205)
  (clean shot296)
  (empty shaker205)
  (empty shot296)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker205 l0)
  (shaker-level shaker205 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail5 ingredient457)
  (cocktail-part2 cocktail5 ingredient437)
)
 (:goal
  (and
      (contains shot296 cocktail5)
)))
