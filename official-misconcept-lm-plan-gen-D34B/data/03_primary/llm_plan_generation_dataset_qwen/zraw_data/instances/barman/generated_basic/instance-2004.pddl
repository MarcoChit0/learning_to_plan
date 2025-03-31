(define (problem prob)
 (:domain barman)
 (:objects 
      shaker84 - shaker
      left right - hand
      shot55 shot75 shot98 - shot
      ingredient305 ingredient327 ingredient285 - ingredient
      cocktail481 - cocktail
      dispenser21 dispenser353 dispenser9 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker84)
  (ontable shot55)
  (ontable shot75)
  (ontable shot98)
  (dispenses dispenser21 ingredient305)
  (dispenses dispenser353 ingredient327)
  (dispenses dispenser9 ingredient285)
  (clean shaker84)
  (clean shot55)
  (clean shot75)
  (clean shot98)
  (empty shaker84)
  (empty shot55)
  (empty shot75)
  (empty shot98)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker84 l0)
  (shaker-level shaker84 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail481 ingredient327)
  (cocktail-part2 cocktail481 ingredient305)
)
 (:goal
  (and
      (contains shot55 cocktail481)
      (contains shot75 ingredient305)
)))
