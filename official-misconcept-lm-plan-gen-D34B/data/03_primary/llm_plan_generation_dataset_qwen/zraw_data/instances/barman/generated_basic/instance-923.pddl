(define (problem prob)
 (:domain barman)
 (:objects 
      shaker418 - shaker
      left right - hand
      shot355 shot46 - shot
      ingredient55 ingredient297 ingredient321 - ingredient
      cocktail1 - cocktail
      dispenser478 dispenser382 dispenser138 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker418)
  (ontable shot355)
  (ontable shot46)
  (dispenses dispenser478 ingredient55)
  (dispenses dispenser382 ingredient297)
  (dispenses dispenser138 ingredient321)
  (clean shaker418)
  (clean shot355)
  (clean shot46)
  (empty shaker418)
  (empty shot355)
  (empty shot46)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker418 l0)
  (shaker-level shaker418 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient297)
  (cocktail-part2 cocktail1 ingredient321)
)
 (:goal
  (and
      (contains shot355 cocktail1)
)))
