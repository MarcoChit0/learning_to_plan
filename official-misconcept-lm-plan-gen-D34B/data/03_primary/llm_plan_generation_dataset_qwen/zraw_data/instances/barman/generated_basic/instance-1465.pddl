(define (problem prob)
 (:domain barman)
 (:objects 
      shaker257 - shaker
      left right - hand
      shot334 shot29 shot208 - shot
      ingredient269 ingredient379 - ingredient
      cocktail359 - cocktail
      dispenser490 dispenser474 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker257)
  (ontable shot334)
  (ontable shot29)
  (ontable shot208)
  (dispenses dispenser490 ingredient269)
  (dispenses dispenser474 ingredient379)
  (clean shaker257)
  (clean shot334)
  (clean shot29)
  (clean shot208)
  (empty shaker257)
  (empty shot334)
  (empty shot29)
  (empty shot208)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker257 l0)
  (shaker-level shaker257 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail359 ingredient379)
  (cocktail-part2 cocktail359 ingredient269)
)
 (:goal
  (and
      (contains shot334 cocktail359)
      (contains shot29 cocktail359)
)))
