(define (problem prob)
 (:domain barman)
 (:objects 
      shaker401 - shaker
      left right - hand
      shot31 shot305 shot202 - shot
      ingredient378 ingredient375 ingredient112 ingredient344 - ingredient
      cocktail195 - cocktail
      dispenser134 dispenser171 dispenser332 dispenser489 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker401)
  (ontable shot31)
  (ontable shot305)
  (ontable shot202)
  (dispenses dispenser134 ingredient378)
  (dispenses dispenser171 ingredient375)
  (dispenses dispenser332 ingredient112)
  (dispenses dispenser489 ingredient344)
  (clean shaker401)
  (clean shot31)
  (clean shot305)
  (clean shot202)
  (empty shaker401)
  (empty shot31)
  (empty shot305)
  (empty shot202)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker401 l0)
  (shaker-level shaker401 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail195 ingredient112)
  (cocktail-part2 cocktail195 ingredient375)
)
 (:goal
  (and
      (contains shot31 cocktail195)
      (contains shot305 ingredient112)
)))
