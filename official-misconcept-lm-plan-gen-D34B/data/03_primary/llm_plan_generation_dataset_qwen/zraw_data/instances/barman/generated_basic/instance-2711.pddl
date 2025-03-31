(define (problem prob)
 (:domain barman)
 (:objects 
      shaker255 - shaker
      left right - hand
      shot338 shot320 shot141 - shot
      ingredient465 ingredient232 - ingredient
      cocktail88 - cocktail
      dispenser421 dispenser477 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker255)
  (ontable shot338)
  (ontable shot320)
  (ontable shot141)
  (dispenses dispenser421 ingredient465)
  (dispenses dispenser477 ingredient232)
  (clean shaker255)
  (clean shot338)
  (clean shot320)
  (clean shot141)
  (empty shaker255)
  (empty shot338)
  (empty shot320)
  (empty shot141)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker255 l0)
  (shaker-level shaker255 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail88 ingredient232)
  (cocktail-part2 cocktail88 ingredient465)
)
 (:goal
  (and
      (contains shot338 cocktail88)
      (contains shot320 cocktail88)
)))
