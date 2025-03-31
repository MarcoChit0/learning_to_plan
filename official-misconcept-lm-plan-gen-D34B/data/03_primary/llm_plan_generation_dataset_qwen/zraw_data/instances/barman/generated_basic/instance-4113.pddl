(define (problem prob)
 (:domain barman)
 (:objects 
      shaker41 - shaker
      left right - hand
      shot436 shot338 - shot
      ingredient480 ingredient453 - ingredient
      cocktail125 - cocktail
      dispenser298 dispenser201 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker41)
  (ontable shot436)
  (ontable shot338)
  (dispenses dispenser298 ingredient480)
  (dispenses dispenser201 ingredient453)
  (clean shaker41)
  (clean shot436)
  (clean shot338)
  (empty shaker41)
  (empty shot436)
  (empty shot338)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker41 l0)
  (shaker-level shaker41 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail125 ingredient453)
  (cocktail-part2 cocktail125 ingredient480)
)
 (:goal
  (and
      (contains shot436 cocktail125)
)))
