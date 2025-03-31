(define (problem prob)
 (:domain barman)
 (:objects 
      shaker224 - shaker
      left right - hand
      shot74 shot241 - shot
      ingredient296 ingredient305 - ingredient
      cocktail1 - cocktail
      dispenser477 dispenser301 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker224)
  (ontable shot74)
  (ontable shot241)
  (dispenses dispenser477 ingredient296)
  (dispenses dispenser301 ingredient305)
  (clean shaker224)
  (clean shot74)
  (clean shot241)
  (empty shaker224)
  (empty shot74)
  (empty shot241)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker224 l0)
  (shaker-level shaker224 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient296)
  (cocktail-part2 cocktail1 ingredient305)
)
 (:goal
  (and
      (contains shot74 cocktail1)
)))
