(define (problem prob)
 (:domain barman)
 (:objects 
      shaker375 - shaker
      left right - hand
      shot204 shot104 - shot
      ingredient133 ingredient465 ingredient71 - ingredient
      cocktail81 - cocktail
      dispenser130 dispenser250 dispenser466 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker375)
  (ontable shot204)
  (ontable shot104)
  (dispenses dispenser130 ingredient133)
  (dispenses dispenser250 ingredient465)
  (dispenses dispenser466 ingredient71)
  (clean shaker375)
  (clean shot204)
  (clean shot104)
  (empty shaker375)
  (empty shot204)
  (empty shot104)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker375 l0)
  (shaker-level shaker375 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail81 ingredient465)
  (cocktail-part2 cocktail81 ingredient133)
)
 (:goal
  (and
      (contains shot204 cocktail81)
)))
