(define (problem prob)
 (:domain barman)
 (:objects 
      shaker59 - shaker
      left right - hand
      shot258 - shot
      ingredient63 ingredient123 ingredient308 ingredient266 - ingredient
      cocktail428 - cocktail
      dispenser109 dispenser436 dispenser228 dispenser251 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker59)
  (ontable shot258)
  (dispenses dispenser109 ingredient63)
  (dispenses dispenser436 ingredient123)
  (dispenses dispenser228 ingredient308)
  (dispenses dispenser251 ingredient266)
  (clean shaker59)
  (clean shot258)
  (empty shaker59)
  (empty shot258)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker59 l0)
  (shaker-level shaker59 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail428 ingredient123)
  (cocktail-part2 cocktail428 ingredient308)
)
 (:goal
  (and
      (contains shot258 cocktail428)
)))
