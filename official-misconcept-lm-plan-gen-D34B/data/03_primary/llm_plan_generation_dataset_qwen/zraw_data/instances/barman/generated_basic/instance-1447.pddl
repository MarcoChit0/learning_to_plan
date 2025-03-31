(define (problem prob)
 (:domain barman)
 (:objects 
      shaker45 - shaker
      left right - hand
      shot428 - shot
      ingredient44 ingredient326 ingredient310 - ingredient
      cocktail318 - cocktail
      dispenser243 dispenser105 dispenser346 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker45)
  (ontable shot428)
  (dispenses dispenser243 ingredient44)
  (dispenses dispenser105 ingredient326)
  (dispenses dispenser346 ingredient310)
  (clean shaker45)
  (clean shot428)
  (empty shaker45)
  (empty shot428)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker45 l0)
  (shaker-level shaker45 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail318 ingredient326)
  (cocktail-part2 cocktail318 ingredient310)
)
 (:goal
  (and
      (contains shot428 cocktail318)
)))
