(define (problem prob)
 (:domain barman)
 (:objects 
      shaker158 - shaker
      left right - hand
      shot87 shot375 - shot
      ingredient392 ingredient67 ingredient9 - ingredient
      cocktail470 - cocktail
      dispenser138 dispenser378 dispenser140 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker158)
  (ontable shot87)
  (ontable shot375)
  (dispenses dispenser138 ingredient392)
  (dispenses dispenser378 ingredient67)
  (dispenses dispenser140 ingredient9)
  (clean shaker158)
  (clean shot87)
  (clean shot375)
  (empty shaker158)
  (empty shot87)
  (empty shot375)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker158 l0)
  (shaker-level shaker158 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail470 ingredient67)
  (cocktail-part2 cocktail470 ingredient392)
)
 (:goal
  (and
      (contains shot87 cocktail470)
)))
