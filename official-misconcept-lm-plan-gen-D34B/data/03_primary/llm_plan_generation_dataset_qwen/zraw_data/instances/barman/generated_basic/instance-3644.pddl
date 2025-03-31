(define (problem prob)
 (:domain barman)
 (:objects 
      shaker104 - shaker
      left right - hand
      shot494 - shot
      ingredient50 ingredient250 ingredient324 ingredient431 - ingredient
      cocktail310 - cocktail
      dispenser287 dispenser447 dispenser136 dispenser385 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker104)
  (ontable shot494)
  (dispenses dispenser287 ingredient50)
  (dispenses dispenser447 ingredient250)
  (dispenses dispenser136 ingredient324)
  (dispenses dispenser385 ingredient431)
  (clean shaker104)
  (clean shot494)
  (empty shaker104)
  (empty shot494)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker104 l0)
  (shaker-level shaker104 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail310 ingredient50)
  (cocktail-part2 cocktail310 ingredient250)
)
 (:goal
  (and
      (contains shot494 cocktail310)
)))
