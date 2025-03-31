(define (problem prob)
 (:domain barman)
 (:objects 
      shaker99 - shaker
      left right - hand
      shot33 - shot
      ingredient32 ingredient87 - ingredient
      cocktail165 - cocktail
      dispenser439 dispenser141 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker99)
  (ontable shot33)
  (dispenses dispenser439 ingredient32)
  (dispenses dispenser141 ingredient87)
  (clean shaker99)
  (clean shot33)
  (empty shaker99)
  (empty shot33)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker99 l0)
  (shaker-level shaker99 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail165 ingredient32)
  (cocktail-part2 cocktail165 ingredient87)
)
 (:goal
  (and
      (contains shot33 cocktail165)
)))
