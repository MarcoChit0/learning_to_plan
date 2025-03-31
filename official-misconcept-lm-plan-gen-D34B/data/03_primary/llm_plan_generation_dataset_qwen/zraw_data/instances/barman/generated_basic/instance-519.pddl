(define (problem prob)
 (:domain barman)
 (:objects 
      shaker313 - shaker
      left right - hand
      shot14 shot284 - shot
      ingredient365 ingredient454 - ingredient
      cocktail1 - cocktail
      dispenser170 dispenser224 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker313)
  (ontable shot14)
  (ontable shot284)
  (dispenses dispenser170 ingredient365)
  (dispenses dispenser224 ingredient454)
  (clean shaker313)
  (clean shot14)
  (clean shot284)
  (empty shaker313)
  (empty shot14)
  (empty shot284)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker313 l0)
  (shaker-level shaker313 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient365)
  (cocktail-part2 cocktail1 ingredient454)
)
 (:goal
  (and
      (contains shot14 cocktail1)
)))
