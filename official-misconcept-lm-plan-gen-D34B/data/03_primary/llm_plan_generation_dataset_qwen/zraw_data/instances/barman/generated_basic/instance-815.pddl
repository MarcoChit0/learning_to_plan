(define (problem prob)
 (:domain barman)
 (:objects 
      shaker350 - shaker
      left right - hand
      shot175 shot484 - shot
      ingredient301 ingredient392 ingredient353 ingredient316 - ingredient
      cocktail1 - cocktail
      dispenser324 dispenser355 dispenser486 dispenser389 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker350)
  (ontable shot175)
  (ontable shot484)
  (dispenses dispenser324 ingredient301)
  (dispenses dispenser355 ingredient392)
  (dispenses dispenser486 ingredient353)
  (dispenses dispenser389 ingredient316)
  (clean shaker350)
  (clean shot175)
  (clean shot484)
  (empty shaker350)
  (empty shot175)
  (empty shot484)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker350 l0)
  (shaker-level shaker350 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient301)
  (cocktail-part2 cocktail1 ingredient392)
)
 (:goal
  (and
      (contains shot175 cocktail1)
)))
