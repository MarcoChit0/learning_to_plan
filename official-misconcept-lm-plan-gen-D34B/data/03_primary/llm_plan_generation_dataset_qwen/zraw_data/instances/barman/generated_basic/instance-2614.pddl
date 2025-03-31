(define (problem prob)
 (:domain barman)
 (:objects 
      shaker385 - shaker
      left right - hand
      shot461 shot227 - shot
      ingredient456 ingredient292 ingredient485 - ingredient
      cocktail106 - cocktail
      dispenser452 dispenser481 dispenser17 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker385)
  (ontable shot461)
  (ontable shot227)
  (dispenses dispenser452 ingredient456)
  (dispenses dispenser481 ingredient292)
  (dispenses dispenser17 ingredient485)
  (clean shaker385)
  (clean shot461)
  (clean shot227)
  (empty shaker385)
  (empty shot461)
  (empty shot227)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker385 l0)
  (shaker-level shaker385 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail106 ingredient456)
  (cocktail-part2 cocktail106 ingredient485)
)
 (:goal
  (and
      (contains shot461 cocktail106)
)))
