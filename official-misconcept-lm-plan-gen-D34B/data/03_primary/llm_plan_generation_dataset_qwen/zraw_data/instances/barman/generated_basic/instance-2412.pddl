(define (problem prob)
 (:domain barman)
 (:objects 
      shaker475 - shaker
      left right - hand
      shot204 - shot
      ingredient360 ingredient329 - ingredient
      cocktail288 - cocktail
      dispenser369 dispenser62 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker475)
  (ontable shot204)
  (dispenses dispenser369 ingredient360)
  (dispenses dispenser62 ingredient329)
  (clean shaker475)
  (clean shot204)
  (empty shaker475)
  (empty shot204)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker475 l0)
  (shaker-level shaker475 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail288 ingredient360)
  (cocktail-part2 cocktail288 ingredient329)
)
 (:goal
  (and
      (contains shot204 cocktail288)
)))
