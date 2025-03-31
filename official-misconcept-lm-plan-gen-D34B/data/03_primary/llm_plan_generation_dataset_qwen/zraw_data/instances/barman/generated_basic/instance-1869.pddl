(define (problem prob)
 (:domain barman)
 (:objects 
      shaker476 - shaker
      left right - hand
      shot473 shot21 - shot
      ingredient375 ingredient396 - ingredient
      cocktail187 - cocktail
      dispenser204 dispenser188 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker476)
  (ontable shot473)
  (ontable shot21)
  (dispenses dispenser204 ingredient375)
  (dispenses dispenser188 ingredient396)
  (clean shaker476)
  (clean shot473)
  (clean shot21)
  (empty shaker476)
  (empty shot473)
  (empty shot21)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker476 l0)
  (shaker-level shaker476 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail187 ingredient396)
  (cocktail-part2 cocktail187 ingredient375)
)
 (:goal
  (and
      (contains shot473 cocktail187)
)))
