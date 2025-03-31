(define (problem prob)
 (:domain barman)
 (:objects 
      shaker399 - shaker
      left right - hand
      shot70 shot95 - shot
      ingredient138 ingredient301 - ingredient
      cocktail302 - cocktail
      dispenser152 dispenser166 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker399)
  (ontable shot70)
  (ontable shot95)
  (dispenses dispenser152 ingredient138)
  (dispenses dispenser166 ingredient301)
  (clean shaker399)
  (clean shot70)
  (clean shot95)
  (empty shaker399)
  (empty shot70)
  (empty shot95)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker399 l0)
  (shaker-level shaker399 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail302 ingredient301)
  (cocktail-part2 cocktail302 ingredient138)
)
 (:goal
  (and
      (contains shot70 cocktail302)
)))
