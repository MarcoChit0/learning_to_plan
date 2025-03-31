(define (problem prob)
 (:domain barman)
 (:objects 
      shaker495 - shaker
      left right - hand
      shot253 shot255 shot73 - shot
      ingredient141 ingredient346 ingredient432 ingredient498 - ingredient
      cocktail380 - cocktail
      dispenser74 dispenser100 dispenser65 dispenser386 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker495)
  (ontable shot253)
  (ontable shot255)
  (ontable shot73)
  (dispenses dispenser74 ingredient141)
  (dispenses dispenser100 ingredient346)
  (dispenses dispenser65 ingredient432)
  (dispenses dispenser386 ingredient498)
  (clean shaker495)
  (clean shot253)
  (clean shot255)
  (clean shot73)
  (empty shaker495)
  (empty shot253)
  (empty shot255)
  (empty shot73)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker495 l0)
  (shaker-level shaker495 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail380 ingredient498)
  (cocktail-part2 cocktail380 ingredient346)
)
 (:goal
  (and
      (contains shot253 cocktail380)
      (contains shot255 ingredient498)
)))
