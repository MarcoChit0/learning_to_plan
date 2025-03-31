(define (problem prob)
 (:domain barman)
 (:objects 
      shaker6 - shaker
      left right - hand
      shot100 shot367 shot384 - shot
      ingredient184 ingredient208 - ingredient
      cocktail376 - cocktail
      dispenser135 dispenser82 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker6)
  (ontable shot100)
  (ontable shot367)
  (ontable shot384)
  (dispenses dispenser135 ingredient184)
  (dispenses dispenser82 ingredient208)
  (clean shaker6)
  (clean shot100)
  (clean shot367)
  (clean shot384)
  (empty shaker6)
  (empty shot100)
  (empty shot367)
  (empty shot384)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker6 l0)
  (shaker-level shaker6 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail376 ingredient208)
  (cocktail-part2 cocktail376 ingredient184)
)
 (:goal
  (and
      (contains shot100 cocktail376)
      (contains shot367 ingredient184)
)))
