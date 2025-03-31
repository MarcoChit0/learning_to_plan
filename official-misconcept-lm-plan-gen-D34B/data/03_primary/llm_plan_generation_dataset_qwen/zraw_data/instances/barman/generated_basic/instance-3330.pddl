(define (problem prob)
 (:domain barman)
 (:objects 
      shaker174 - shaker
      left right - hand
      shot400 shot499 - shot
      ingredient412 ingredient279 - ingredient
      cocktail4 - cocktail
      dispenser170 dispenser291 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker174)
  (ontable shot400)
  (ontable shot499)
  (dispenses dispenser170 ingredient412)
  (dispenses dispenser291 ingredient279)
  (clean shaker174)
  (clean shot400)
  (clean shot499)
  (empty shaker174)
  (empty shot400)
  (empty shot499)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker174 l0)
  (shaker-level shaker174 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail4 ingredient412)
  (cocktail-part2 cocktail4 ingredient279)
)
 (:goal
  (and
      (contains shot400 cocktail4)
)))
