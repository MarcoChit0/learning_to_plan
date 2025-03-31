(define (problem prob)
 (:domain barman)
 (:objects 
      shaker24 - shaker
      left right - hand
      shot326 shot253 - shot
      ingredient223 ingredient60 ingredient416 ingredient273 - ingredient
      cocktail212 - cocktail
      dispenser141 dispenser119 dispenser152 dispenser309 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker24)
  (ontable shot326)
  (ontable shot253)
  (dispenses dispenser141 ingredient223)
  (dispenses dispenser119 ingredient60)
  (dispenses dispenser152 ingredient416)
  (dispenses dispenser309 ingredient273)
  (clean shaker24)
  (clean shot326)
  (clean shot253)
  (empty shaker24)
  (empty shot326)
  (empty shot253)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker24 l0)
  (shaker-level shaker24 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail212 ingredient416)
  (cocktail-part2 cocktail212 ingredient60)
)
 (:goal
  (and
      (contains shot326 cocktail212)
)))
