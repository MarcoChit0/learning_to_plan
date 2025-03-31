(define (problem prob)
 (:domain barman)
 (:objects 
      shaker490 - shaker
      left right - hand
      shot124 - shot
      ingredient469 ingredient259 ingredient458 ingredient219 - ingredient
      cocktail71 - cocktail
      dispenser211 dispenser164 dispenser430 dispenser474 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker490)
  (ontable shot124)
  (dispenses dispenser211 ingredient469)
  (dispenses dispenser164 ingredient259)
  (dispenses dispenser430 ingredient458)
  (dispenses dispenser474 ingredient219)
  (clean shaker490)
  (clean shot124)
  (empty shaker490)
  (empty shot124)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker490 l0)
  (shaker-level shaker490 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail71 ingredient458)
  (cocktail-part2 cocktail71 ingredient469)
)
 (:goal
  (and
      (contains shot124 cocktail71)
)))
