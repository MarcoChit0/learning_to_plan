(define (problem prob)
 (:domain barman)
 (:objects 
      shaker364 - shaker
      left right - hand
      shot95 shot209 - shot
      ingredient267 ingredient32 ingredient290 ingredient171 - ingredient
      cocktail489 - cocktail
      dispenser344 dispenser189 dispenser444 dispenser111 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker364)
  (ontable shot95)
  (ontable shot209)
  (dispenses dispenser344 ingredient267)
  (dispenses dispenser189 ingredient32)
  (dispenses dispenser444 ingredient290)
  (dispenses dispenser111 ingredient171)
  (clean shaker364)
  (clean shot95)
  (clean shot209)
  (empty shaker364)
  (empty shot95)
  (empty shot209)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker364 l0)
  (shaker-level shaker364 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail489 ingredient32)
  (cocktail-part2 cocktail489 ingredient290)
)
 (:goal
  (and
      (contains shot95 cocktail489)
)))
