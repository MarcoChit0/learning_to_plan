(define (problem prob)
 (:domain barman)
 (:objects 
      shaker213 - shaker
      left right - hand
      shot55 - shot
      ingredient111 ingredient123 ingredient308 - ingredient
      cocktail188 - cocktail
      dispenser283 dispenser31 dispenser60 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker213)
  (ontable shot55)
  (dispenses dispenser283 ingredient111)
  (dispenses dispenser31 ingredient123)
  (dispenses dispenser60 ingredient308)
  (clean shaker213)
  (clean shot55)
  (empty shaker213)
  (empty shot55)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker213 l0)
  (shaker-level shaker213 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail188 ingredient308)
  (cocktail-part2 cocktail188 ingredient123)
)
 (:goal
  (and
      (contains shot55 cocktail188)
)))
