(define (problem prob)
 (:domain barman)
 (:objects 
      shaker490 - shaker
      left right - hand
      shot157 - shot
      ingredient88 ingredient128 - ingredient
      cocktail253 - cocktail
      dispenser372 dispenser313 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker490)
  (ontable shot157)
  (dispenses dispenser372 ingredient88)
  (dispenses dispenser313 ingredient128)
  (clean shaker490)
  (clean shot157)
  (empty shaker490)
  (empty shot157)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker490 l0)
  (shaker-level shaker490 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail253 ingredient88)
  (cocktail-part2 cocktail253 ingredient128)
)
 (:goal
  (and
      (contains shot157 cocktail253)
)))
