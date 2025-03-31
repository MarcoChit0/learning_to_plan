(define (problem prob)
 (:domain barman)
 (:objects 
      shaker26 - shaker
      left right - hand
      shot246 shot87 - shot
      ingredient444 ingredient384 ingredient74 - ingredient
      cocktail333 - cocktail
      dispenser113 dispenser236 dispenser161 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker26)
  (ontable shot246)
  (ontable shot87)
  (dispenses dispenser113 ingredient444)
  (dispenses dispenser236 ingredient384)
  (dispenses dispenser161 ingredient74)
  (clean shaker26)
  (clean shot246)
  (clean shot87)
  (empty shaker26)
  (empty shot246)
  (empty shot87)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker26 l0)
  (shaker-level shaker26 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail333 ingredient74)
  (cocktail-part2 cocktail333 ingredient384)
)
 (:goal
  (and
      (contains shot246 cocktail333)
)))
