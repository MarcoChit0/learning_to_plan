(define (problem prob)
 (:domain barman)
 (:objects 
      shaker428 - shaker
      left right - hand
      shot48 - shot
      ingredient65 ingredient192 - ingredient
      cocktail1 - cocktail
      dispenser46 dispenser428 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker428)
  (ontable shot48)
  (dispenses dispenser46 ingredient65)
  (dispenses dispenser428 ingredient192)
  (clean shaker428)
  (clean shot48)
  (empty shaker428)
  (empty shot48)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker428 l0)
  (shaker-level shaker428 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient192)
  (cocktail-part2 cocktail1 ingredient65)
)
 (:goal
  (and
      (contains shot48 cocktail1)
)))
