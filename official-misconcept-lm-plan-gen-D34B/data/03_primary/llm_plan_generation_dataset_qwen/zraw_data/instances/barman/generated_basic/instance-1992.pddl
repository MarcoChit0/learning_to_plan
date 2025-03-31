(define (problem prob)
 (:domain barman)
 (:objects 
      shaker325 - shaker
      left right - hand
      shot375 shot182 - shot
      ingredient250 ingredient22 ingredient258 - ingredient
      cocktail425 - cocktail
      dispenser426 dispenser172 dispenser434 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker325)
  (ontable shot375)
  (ontable shot182)
  (dispenses dispenser426 ingredient250)
  (dispenses dispenser172 ingredient22)
  (dispenses dispenser434 ingredient258)
  (clean shaker325)
  (clean shot375)
  (clean shot182)
  (empty shaker325)
  (empty shot375)
  (empty shot182)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker325 l0)
  (shaker-level shaker325 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail425 ingredient258)
  (cocktail-part2 cocktail425 ingredient250)
)
 (:goal
  (and
      (contains shot375 cocktail425)
)))
