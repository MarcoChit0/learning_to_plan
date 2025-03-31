(define (problem prob)
 (:domain barman)
 (:objects 
      shaker233 - shaker
      left right - hand
      shot0 shot137 - shot
      ingredient182 ingredient246 - ingredient
      cocktail1 - cocktail
      dispenser4 dispenser26 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker233)
  (ontable shot0)
  (ontable shot137)
  (dispenses dispenser4 ingredient182)
  (dispenses dispenser26 ingredient246)
  (clean shaker233)
  (clean shot0)
  (clean shot137)
  (empty shaker233)
  (empty shot0)
  (empty shot137)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker233 l0)
  (shaker-level shaker233 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient182)
  (cocktail-part2 cocktail1 ingredient246)
)
 (:goal
  (and
      (contains shot0 cocktail1)
)))
