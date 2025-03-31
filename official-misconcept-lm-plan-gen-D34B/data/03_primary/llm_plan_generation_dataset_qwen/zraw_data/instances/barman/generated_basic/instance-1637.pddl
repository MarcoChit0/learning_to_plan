(define (problem prob)
 (:domain barman)
 (:objects 
      shaker355 - shaker
      left right - hand
      shot290 shot299 shot379 - shot
      ingredient439 ingredient350 - ingredient
      cocktail166 - cocktail
      dispenser289 dispenser90 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker355)
  (ontable shot290)
  (ontable shot299)
  (ontable shot379)
  (dispenses dispenser289 ingredient439)
  (dispenses dispenser90 ingredient350)
  (clean shaker355)
  (clean shot290)
  (clean shot299)
  (clean shot379)
  (empty shaker355)
  (empty shot290)
  (empty shot299)
  (empty shot379)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker355 l0)
  (shaker-level shaker355 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail166 ingredient350)
  (cocktail-part2 cocktail166 ingredient439)
)
 (:goal
  (and
      (contains shot290 cocktail166)
      (contains shot299 cocktail166)
)))
