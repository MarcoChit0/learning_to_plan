(define (problem prob)
 (:domain barman)
 (:objects 
      shaker273 - shaker
      left right - hand
      shot365 shot255 - shot
      ingredient410 ingredient96 - ingredient
      cocktail1 - cocktail
      dispenser285 dispenser228 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker273)
  (ontable shot365)
  (ontable shot255)
  (dispenses dispenser285 ingredient410)
  (dispenses dispenser228 ingredient96)
  (clean shaker273)
  (clean shot365)
  (clean shot255)
  (empty shaker273)
  (empty shot365)
  (empty shot255)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker273 l0)
  (shaker-level shaker273 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient96)
  (cocktail-part2 cocktail1 ingredient410)
)
 (:goal
  (and
      (contains shot365 cocktail1)
)))
