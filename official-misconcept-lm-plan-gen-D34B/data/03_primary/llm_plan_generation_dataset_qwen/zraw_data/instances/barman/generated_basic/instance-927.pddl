(define (problem prob)
 (:domain barman)
 (:objects 
      shaker454 - shaker
      left right - hand
      shot363 shot153 - shot
      ingredient33 ingredient395 ingredient277 - ingredient
      cocktail1 - cocktail
      dispenser162 dispenser166 dispenser395 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker454)
  (ontable shot363)
  (ontable shot153)
  (dispenses dispenser162 ingredient33)
  (dispenses dispenser166 ingredient395)
  (dispenses dispenser395 ingredient277)
  (clean shaker454)
  (clean shot363)
  (clean shot153)
  (empty shaker454)
  (empty shot363)
  (empty shot153)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker454 l0)
  (shaker-level shaker454 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient33)
  (cocktail-part2 cocktail1 ingredient277)
)
 (:goal
  (and
      (contains shot363 cocktail1)
)))
