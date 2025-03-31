(define (problem prob)
 (:domain barman)
 (:objects 
      shaker187 - shaker
      left right - hand
      shot128 - shot
      ingredient69 ingredient186 ingredient316 ingredient133 - ingredient
      cocktail1 - cocktail
      dispenser201 dispenser71 dispenser108 dispenser20 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker187)
  (ontable shot128)
  (dispenses dispenser201 ingredient69)
  (dispenses dispenser71 ingredient186)
  (dispenses dispenser108 ingredient316)
  (dispenses dispenser20 ingredient133)
  (clean shaker187)
  (clean shot128)
  (empty shaker187)
  (empty shot128)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker187 l0)
  (shaker-level shaker187 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient69)
  (cocktail-part2 cocktail1 ingredient133)
)
 (:goal
  (and
      (contains shot128 cocktail1)
)))
