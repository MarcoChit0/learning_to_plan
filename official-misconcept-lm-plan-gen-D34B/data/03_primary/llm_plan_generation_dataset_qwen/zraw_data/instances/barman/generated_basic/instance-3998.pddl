(define (problem prob)
 (:domain barman)
 (:objects 
      shaker26 - shaker
      left right - hand
      shot386 shot89 shot460 - shot
      ingredient27 ingredient18 - ingredient
      cocktail308 - cocktail
      dispenser449 dispenser93 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker26)
  (ontable shot386)
  (ontable shot89)
  (ontable shot460)
  (dispenses dispenser449 ingredient27)
  (dispenses dispenser93 ingredient18)
  (clean shaker26)
  (clean shot386)
  (clean shot89)
  (clean shot460)
  (empty shaker26)
  (empty shot386)
  (empty shot89)
  (empty shot460)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker26 l0)
  (shaker-level shaker26 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail308 ingredient27)
  (cocktail-part2 cocktail308 ingredient18)
)
 (:goal
  (and
      (contains shot386 cocktail308)
      (contains shot89 cocktail308)
)))
