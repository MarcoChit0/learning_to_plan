(define (problem prob)
 (:domain barman)
 (:objects 
      shaker191 - shaker
      left right - hand
      shot403 shot245 shot443 - shot
      ingredient244 ingredient365 ingredient388 ingredient452 - ingredient
      cocktail70 - cocktail
      dispenser208 dispenser481 dispenser378 dispenser131 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker191)
  (ontable shot403)
  (ontable shot245)
  (ontable shot443)
  (dispenses dispenser208 ingredient244)
  (dispenses dispenser481 ingredient365)
  (dispenses dispenser378 ingredient388)
  (dispenses dispenser131 ingredient452)
  (clean shaker191)
  (clean shot403)
  (clean shot245)
  (clean shot443)
  (empty shaker191)
  (empty shot403)
  (empty shot245)
  (empty shot443)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker191 l0)
  (shaker-level shaker191 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail70 ingredient388)
  (cocktail-part2 cocktail70 ingredient244)
)
 (:goal
  (and
      (contains shot403 cocktail70)
      (contains shot245 ingredient365)
)))
