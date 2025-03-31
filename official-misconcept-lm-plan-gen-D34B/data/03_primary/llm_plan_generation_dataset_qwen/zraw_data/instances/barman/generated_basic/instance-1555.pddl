(define (problem prob)
 (:domain barman)
 (:objects 
      shaker288 - shaker
      left right - hand
      shot317 - shot
      ingredient209 ingredient92 - ingredient
      cocktail370 - cocktail
      dispenser290 dispenser494 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker288)
  (ontable shot317)
  (dispenses dispenser290 ingredient209)
  (dispenses dispenser494 ingredient92)
  (clean shaker288)
  (clean shot317)
  (empty shaker288)
  (empty shot317)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker288 l0)
  (shaker-level shaker288 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail370 ingredient209)
  (cocktail-part2 cocktail370 ingredient92)
)
 (:goal
  (and
      (contains shot317 cocktail370)
)))
