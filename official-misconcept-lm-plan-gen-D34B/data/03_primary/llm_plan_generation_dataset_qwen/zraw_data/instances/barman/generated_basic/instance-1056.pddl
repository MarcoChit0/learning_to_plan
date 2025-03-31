(define (problem prob)
 (:domain barman)
 (:objects 
      shaker48 - shaker
      left right - hand
      shot102 shot478 shot224 - shot
      ingredient318 ingredient301 ingredient41 ingredient432 - ingredient
      cocktail1 - cocktail
      dispenser201 dispenser365 dispenser275 dispenser30 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker48)
  (ontable shot102)
  (ontable shot478)
  (ontable shot224)
  (dispenses dispenser201 ingredient318)
  (dispenses dispenser365 ingredient301)
  (dispenses dispenser275 ingredient41)
  (dispenses dispenser30 ingredient432)
  (clean shaker48)
  (clean shot102)
  (clean shot478)
  (clean shot224)
  (empty shaker48)
  (empty shot102)
  (empty shot478)
  (empty shot224)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker48 l0)
  (shaker-level shaker48 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient41)
  (cocktail-part2 cocktail1 ingredient301)
)
 (:goal
  (and
      (contains shot102 cocktail1)
      (contains shot478 cocktail1)
)))
