(define (problem prob)
 (:domain barman)
 (:objects 
      shaker451 - shaker
      left right - hand
      shot259 shot374 - shot
      ingredient180 ingredient483 ingredient60 - ingredient
      cocktail355 - cocktail
      dispenser237 dispenser411 dispenser469 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker451)
  (ontable shot259)
  (ontable shot374)
  (dispenses dispenser237 ingredient180)
  (dispenses dispenser411 ingredient483)
  (dispenses dispenser469 ingredient60)
  (clean shaker451)
  (clean shot259)
  (clean shot374)
  (empty shaker451)
  (empty shot259)
  (empty shot374)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker451 l0)
  (shaker-level shaker451 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail355 ingredient180)
  (cocktail-part2 cocktail355 ingredient483)
)
 (:goal
  (and
      (contains shot259 cocktail355)
)))
