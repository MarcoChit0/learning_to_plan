(define (problem prob)
 (:domain barman)
 (:objects 
      shaker90 - shaker
      left right - hand
      shot369 shot174 - shot
      ingredient376 ingredient147 ingredient301 - ingredient
      cocktail214 - cocktail
      dispenser456 dispenser103 dispenser231 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker90)
  (ontable shot369)
  (ontable shot174)
  (dispenses dispenser456 ingredient376)
  (dispenses dispenser103 ingredient147)
  (dispenses dispenser231 ingredient301)
  (clean shaker90)
  (clean shot369)
  (clean shot174)
  (empty shaker90)
  (empty shot369)
  (empty shot174)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker90 l0)
  (shaker-level shaker90 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail214 ingredient376)
  (cocktail-part2 cocktail214 ingredient301)
)
 (:goal
  (and
      (contains shot369 cocktail214)
)))
