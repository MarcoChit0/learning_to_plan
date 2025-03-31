(define (problem prob)
 (:domain barman)
 (:objects 
      shaker257 - shaker
      left right - hand
      shot434 shot329 - shot
      ingredient388 ingredient224 - ingredient
      cocktail208 - cocktail
      dispenser382 dispenser199 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker257)
  (ontable shot434)
  (ontable shot329)
  (dispenses dispenser382 ingredient388)
  (dispenses dispenser199 ingredient224)
  (clean shaker257)
  (clean shot434)
  (clean shot329)
  (empty shaker257)
  (empty shot434)
  (empty shot329)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker257 l0)
  (shaker-level shaker257 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail208 ingredient388)
  (cocktail-part2 cocktail208 ingredient224)
)
 (:goal
  (and
      (contains shot434 cocktail208)
)))
