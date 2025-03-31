(define (problem prob)
 (:domain barman)
 (:objects 
      shaker450 - shaker
      left right - hand
      shot222 shot484 - shot
      ingredient442 ingredient498 - ingredient
      cocktail1 - cocktail
      dispenser3 dispenser394 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker450)
  (ontable shot222)
  (ontable shot484)
  (dispenses dispenser3 ingredient442)
  (dispenses dispenser394 ingredient498)
  (clean shaker450)
  (clean shot222)
  (clean shot484)
  (empty shaker450)
  (empty shot222)
  (empty shot484)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker450 l0)
  (shaker-level shaker450 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient498)
  (cocktail-part2 cocktail1 ingredient442)
)
 (:goal
  (and
      (contains shot222 cocktail1)
)))
