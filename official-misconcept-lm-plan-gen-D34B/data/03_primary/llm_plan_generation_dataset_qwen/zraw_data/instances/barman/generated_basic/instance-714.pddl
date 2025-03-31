(define (problem prob)
 (:domain barman)
 (:objects 
      shaker465 - shaker
      left right - hand
      shot303 shot378 shot366 - shot
      ingredient352 ingredient61 ingredient130 ingredient386 - ingredient
      cocktail1 - cocktail
      dispenser156 dispenser485 dispenser499 dispenser288 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker465)
  (ontable shot303)
  (ontable shot378)
  (ontable shot366)
  (dispenses dispenser156 ingredient352)
  (dispenses dispenser485 ingredient61)
  (dispenses dispenser499 ingredient130)
  (dispenses dispenser288 ingredient386)
  (clean shaker465)
  (clean shot303)
  (clean shot378)
  (clean shot366)
  (empty shaker465)
  (empty shot303)
  (empty shot378)
  (empty shot366)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker465 l0)
  (shaker-level shaker465 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient130)
  (cocktail-part2 cocktail1 ingredient352)
)
 (:goal
  (and
      (contains shot303 cocktail1)
      (contains shot378 cocktail1)
)))
