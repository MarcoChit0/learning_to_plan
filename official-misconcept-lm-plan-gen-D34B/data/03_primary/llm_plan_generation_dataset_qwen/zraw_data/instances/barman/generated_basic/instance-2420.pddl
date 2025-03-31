(define (problem prob)
 (:domain barman)
 (:objects 
      shaker336 - shaker
      left right - hand
      shot135 - shot
      ingredient88 ingredient77 - ingredient
      cocktail414 - cocktail
      dispenser1 dispenser496 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker336)
  (ontable shot135)
  (dispenses dispenser1 ingredient88)
  (dispenses dispenser496 ingredient77)
  (clean shaker336)
  (clean shot135)
  (empty shaker336)
  (empty shot135)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker336 l0)
  (shaker-level shaker336 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail414 ingredient77)
  (cocktail-part2 cocktail414 ingredient88)
)
 (:goal
  (and
      (contains shot135 cocktail414)
)))
