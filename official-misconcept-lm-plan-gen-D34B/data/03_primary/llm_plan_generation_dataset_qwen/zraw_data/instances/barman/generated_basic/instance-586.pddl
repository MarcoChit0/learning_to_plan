(define (problem prob)
 (:domain barman)
 (:objects 
      shaker180 - shaker
      left right - hand
      shot466 shot29 - shot
      ingredient43 ingredient77 - ingredient
      cocktail1 - cocktail
      dispenser454 dispenser52 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker180)
  (ontable shot466)
  (ontable shot29)
  (dispenses dispenser454 ingredient43)
  (dispenses dispenser52 ingredient77)
  (clean shaker180)
  (clean shot466)
  (clean shot29)
  (empty shaker180)
  (empty shot466)
  (empty shot29)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker180 l0)
  (shaker-level shaker180 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient77)
  (cocktail-part2 cocktail1 ingredient43)
)
 (:goal
  (and
      (contains shot466 cocktail1)
)))
