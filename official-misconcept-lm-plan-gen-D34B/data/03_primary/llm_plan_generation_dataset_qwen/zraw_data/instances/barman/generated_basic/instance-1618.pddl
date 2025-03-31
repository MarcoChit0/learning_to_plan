(define (problem prob)
 (:domain barman)
 (:objects 
      shaker499 - shaker
      left right - hand
      shot395 shot476 - shot
      ingredient398 ingredient473 ingredient333 - ingredient
      cocktail140 - cocktail
      dispenser4 dispenser174 dispenser40 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker499)
  (ontable shot395)
  (ontable shot476)
  (dispenses dispenser4 ingredient398)
  (dispenses dispenser174 ingredient473)
  (dispenses dispenser40 ingredient333)
  (clean shaker499)
  (clean shot395)
  (clean shot476)
  (empty shaker499)
  (empty shot395)
  (empty shot476)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker499 l0)
  (shaker-level shaker499 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail140 ingredient473)
  (cocktail-part2 cocktail140 ingredient398)
)
 (:goal
  (and
      (contains shot395 cocktail140)
)))
