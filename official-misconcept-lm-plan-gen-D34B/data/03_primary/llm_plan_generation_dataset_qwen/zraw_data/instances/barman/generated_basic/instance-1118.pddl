(define (problem prob)
 (:domain barman)
 (:objects 
      shaker257 - shaker
      left right - hand
      shot428 shot186 - shot
      ingredient408 ingredient259 - ingredient
      cocktail1 - cocktail
      dispenser303 dispenser276 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker257)
  (ontable shot428)
  (ontable shot186)
  (dispenses dispenser303 ingredient408)
  (dispenses dispenser276 ingredient259)
  (clean shaker257)
  (clean shot428)
  (clean shot186)
  (empty shaker257)
  (empty shot428)
  (empty shot186)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker257 l0)
  (shaker-level shaker257 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient259)
  (cocktail-part2 cocktail1 ingredient408)
)
 (:goal
  (and
      (contains shot428 cocktail1)
)))
