(define (problem prob)
 (:domain barman)
 (:objects 
      shaker286 - shaker
      left right - hand
      shot99 - shot
      ingredient471 ingredient352 ingredient426 ingredient47 - ingredient
      cocktail212 - cocktail
      dispenser176 dispenser346 dispenser303 dispenser444 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker286)
  (ontable shot99)
  (dispenses dispenser176 ingredient471)
  (dispenses dispenser346 ingredient352)
  (dispenses dispenser303 ingredient426)
  (dispenses dispenser444 ingredient47)
  (clean shaker286)
  (clean shot99)
  (empty shaker286)
  (empty shot99)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker286 l0)
  (shaker-level shaker286 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail212 ingredient352)
  (cocktail-part2 cocktail212 ingredient471)
)
 (:goal
  (and
      (contains shot99 cocktail212)
)))
