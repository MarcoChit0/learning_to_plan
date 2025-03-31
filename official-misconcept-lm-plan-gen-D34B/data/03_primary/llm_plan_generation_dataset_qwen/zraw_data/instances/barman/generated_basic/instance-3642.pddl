(define (problem prob)
 (:domain barman)
 (:objects 
      shaker284 - shaker
      left right - hand
      shot15 - shot
      ingredient336 ingredient484 ingredient260 ingredient174 - ingredient
      cocktail256 - cocktail
      dispenser334 dispenser68 dispenser435 dispenser336 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker284)
  (ontable shot15)
  (dispenses dispenser334 ingredient336)
  (dispenses dispenser68 ingredient484)
  (dispenses dispenser435 ingredient260)
  (dispenses dispenser336 ingredient174)
  (clean shaker284)
  (clean shot15)
  (empty shaker284)
  (empty shot15)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker284 l0)
  (shaker-level shaker284 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail256 ingredient484)
  (cocktail-part2 cocktail256 ingredient336)
)
 (:goal
  (and
      (contains shot15 cocktail256)
)))
