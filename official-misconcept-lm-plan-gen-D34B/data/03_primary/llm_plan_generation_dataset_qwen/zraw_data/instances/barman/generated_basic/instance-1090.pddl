(define (problem prob)
 (:domain barman)
 (:objects 
      shaker109 - shaker
      left right - hand
      shot303 - shot
      ingredient139 ingredient178 ingredient326 - ingredient
      cocktail1 - cocktail
      dispenser337 dispenser17 dispenser198 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker109)
  (ontable shot303)
  (dispenses dispenser337 ingredient139)
  (dispenses dispenser17 ingredient178)
  (dispenses dispenser198 ingredient326)
  (clean shaker109)
  (clean shot303)
  (empty shaker109)
  (empty shot303)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker109 l0)
  (shaker-level shaker109 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient326)
  (cocktail-part2 cocktail1 ingredient178)
)
 (:goal
  (and
      (contains shot303 cocktail1)
)))
