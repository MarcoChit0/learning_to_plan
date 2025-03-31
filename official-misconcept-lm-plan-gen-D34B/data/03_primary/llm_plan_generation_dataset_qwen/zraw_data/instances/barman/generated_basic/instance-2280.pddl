(define (problem prob)
 (:domain barman)
 (:objects 
      shaker82 - shaker
      left right - hand
      shot378 - shot
      ingredient192 ingredient299 ingredient135 - ingredient
      cocktail384 - cocktail
      dispenser141 dispenser489 dispenser317 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker82)
  (ontable shot378)
  (dispenses dispenser141 ingredient192)
  (dispenses dispenser489 ingredient299)
  (dispenses dispenser317 ingredient135)
  (clean shaker82)
  (clean shot378)
  (empty shaker82)
  (empty shot378)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker82 l0)
  (shaker-level shaker82 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail384 ingredient192)
  (cocktail-part2 cocktail384 ingredient299)
)
 (:goal
  (and
      (contains shot378 cocktail384)
)))
