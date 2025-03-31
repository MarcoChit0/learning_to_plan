(define (problem prob)
 (:domain barman)
 (:objects 
      shaker437 - shaker
      left right - hand
      shot368 shot110 - shot
      ingredient279 ingredient63 - ingredient
      cocktail1 - cocktail
      dispenser424 dispenser423 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker437)
  (ontable shot368)
  (ontable shot110)
  (dispenses dispenser424 ingredient279)
  (dispenses dispenser423 ingredient63)
  (clean shaker437)
  (clean shot368)
  (clean shot110)
  (empty shaker437)
  (empty shot368)
  (empty shot110)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker437 l0)
  (shaker-level shaker437 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient63)
  (cocktail-part2 cocktail1 ingredient279)
)
 (:goal
  (and
      (contains shot368 cocktail1)
)))
