(define (problem prob)
 (:domain barman)
 (:objects 
      shaker301 - shaker
      left right - hand
      shot294 shot215 - shot
      ingredient483 ingredient256 ingredient115 - ingredient
      cocktail34 - cocktail
      dispenser98 dispenser69 dispenser119 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker301)
  (ontable shot294)
  (ontable shot215)
  (dispenses dispenser98 ingredient483)
  (dispenses dispenser69 ingredient256)
  (dispenses dispenser119 ingredient115)
  (clean shaker301)
  (clean shot294)
  (clean shot215)
  (empty shaker301)
  (empty shot294)
  (empty shot215)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker301 l0)
  (shaker-level shaker301 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail34 ingredient483)
  (cocktail-part2 cocktail34 ingredient256)
)
 (:goal
  (and
      (contains shot294 cocktail34)
)))
