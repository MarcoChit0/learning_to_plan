(define (problem prob)
 (:domain barman)
 (:objects 
      shaker194 - shaker
      left right - hand
      shot56 shot8 - shot
      ingredient383 ingredient80 - ingredient
      cocktail129 - cocktail
      dispenser34 dispenser158 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker194)
  (ontable shot56)
  (ontable shot8)
  (dispenses dispenser34 ingredient383)
  (dispenses dispenser158 ingredient80)
  (clean shaker194)
  (clean shot56)
  (clean shot8)
  (empty shaker194)
  (empty shot56)
  (empty shot8)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker194 l0)
  (shaker-level shaker194 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail129 ingredient80)
  (cocktail-part2 cocktail129 ingredient383)
)
 (:goal
  (and
      (contains shot56 cocktail129)
)))
