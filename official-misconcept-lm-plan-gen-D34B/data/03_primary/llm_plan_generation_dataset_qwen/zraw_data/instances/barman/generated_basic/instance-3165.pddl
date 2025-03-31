(define (problem prob)
 (:domain barman)
 (:objects 
      shaker339 - shaker
      left right - hand
      shot432 shot15 shot319 - shot
      ingredient48 ingredient182 ingredient177 ingredient475 - ingredient
      cocktail9 - cocktail
      dispenser468 dispenser389 dispenser411 dispenser207 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker339)
  (ontable shot432)
  (ontable shot15)
  (ontable shot319)
  (dispenses dispenser468 ingredient48)
  (dispenses dispenser389 ingredient182)
  (dispenses dispenser411 ingredient177)
  (dispenses dispenser207 ingredient475)
  (clean shaker339)
  (clean shot432)
  (clean shot15)
  (clean shot319)
  (empty shaker339)
  (empty shot432)
  (empty shot15)
  (empty shot319)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker339 l0)
  (shaker-level shaker339 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail9 ingredient182)
  (cocktail-part2 cocktail9 ingredient177)
)
 (:goal
  (and
      (contains shot432 cocktail9)
      (contains shot15 cocktail9)
)))
