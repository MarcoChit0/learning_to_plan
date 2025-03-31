(define (problem prob)
 (:domain barman)
 (:objects 
      shaker20 - shaker
      left right - hand
      shot461 shot364 shot481 - shot
      ingredient471 ingredient88 - ingredient
      cocktail382 - cocktail
      dispenser233 dispenser409 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker20)
  (ontable shot461)
  (ontable shot364)
  (ontable shot481)
  (dispenses dispenser233 ingredient471)
  (dispenses dispenser409 ingredient88)
  (clean shaker20)
  (clean shot461)
  (clean shot364)
  (clean shot481)
  (empty shaker20)
  (empty shot461)
  (empty shot364)
  (empty shot481)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker20 l0)
  (shaker-level shaker20 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail382 ingredient88)
  (cocktail-part2 cocktail382 ingredient471)
)
 (:goal
  (and
      (contains shot461 cocktail382)
      (contains shot364 ingredient88)
)))
