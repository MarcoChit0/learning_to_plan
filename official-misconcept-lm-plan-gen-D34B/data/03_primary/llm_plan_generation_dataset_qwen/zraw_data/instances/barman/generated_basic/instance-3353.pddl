(define (problem prob)
 (:domain barman)
 (:objects 
      shaker439 - shaker
      left right - hand
      shot76 - shot
      ingredient81 ingredient122 - ingredient
      cocktail152 - cocktail
      dispenser44 dispenser127 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker439)
  (ontable shot76)
  (dispenses dispenser44 ingredient81)
  (dispenses dispenser127 ingredient122)
  (clean shaker439)
  (clean shot76)
  (empty shaker439)
  (empty shot76)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker439 l0)
  (shaker-level shaker439 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail152 ingredient122)
  (cocktail-part2 cocktail152 ingredient81)
)
 (:goal
  (and
      (contains shot76 cocktail152)
)))
