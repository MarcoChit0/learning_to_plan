(define (problem prob)
 (:domain barman)
 (:objects 
      shaker45 - shaker
      left right - hand
      shot379 shot190 shot450 - shot
      ingredient194 ingredient53 ingredient27 ingredient418 - ingredient
      cocktail34 - cocktail
      dispenser93 dispenser409 dispenser5 dispenser304 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker45)
  (ontable shot379)
  (ontable shot190)
  (ontable shot450)
  (dispenses dispenser93 ingredient194)
  (dispenses dispenser409 ingredient53)
  (dispenses dispenser5 ingredient27)
  (dispenses dispenser304 ingredient418)
  (clean shaker45)
  (clean shot379)
  (clean shot190)
  (clean shot450)
  (empty shaker45)
  (empty shot379)
  (empty shot190)
  (empty shot450)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker45 l0)
  (shaker-level shaker45 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail34 ingredient53)
  (cocktail-part2 cocktail34 ingredient27)
)
 (:goal
  (and
      (contains shot379 cocktail34)
      (contains shot190 ingredient418)
)))
