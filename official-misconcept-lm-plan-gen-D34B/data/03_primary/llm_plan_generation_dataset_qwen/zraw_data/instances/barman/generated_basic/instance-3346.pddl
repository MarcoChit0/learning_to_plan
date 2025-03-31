(define (problem prob)
 (:domain barman)
 (:objects 
      shaker413 - shaker
      left right - hand
      shot220 shot346 - shot
      ingredient295 ingredient298 - ingredient
      cocktail304 - cocktail
      dispenser156 dispenser150 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker413)
  (ontable shot220)
  (ontable shot346)
  (dispenses dispenser156 ingredient295)
  (dispenses dispenser150 ingredient298)
  (clean shaker413)
  (clean shot220)
  (clean shot346)
  (empty shaker413)
  (empty shot220)
  (empty shot346)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker413 l0)
  (shaker-level shaker413 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail304 ingredient298)
  (cocktail-part2 cocktail304 ingredient295)
)
 (:goal
  (and
      (contains shot220 cocktail304)
)))
