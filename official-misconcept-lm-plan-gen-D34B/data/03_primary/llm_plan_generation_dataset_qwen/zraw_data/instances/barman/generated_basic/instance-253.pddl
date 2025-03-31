(define (problem prob)
 (:domain barman)
 (:objects 
      shaker391 - shaker
      left right - hand
      shot118 shot348 shot360 - shot
      ingredient12 ingredient237 ingredient171 - ingredient
      cocktail1 - cocktail
      dispenser413 dispenser180 dispenser134 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker391)
  (ontable shot118)
  (ontable shot348)
  (ontable shot360)
  (dispenses dispenser413 ingredient12)
  (dispenses dispenser180 ingredient237)
  (dispenses dispenser134 ingredient171)
  (clean shaker391)
  (clean shot118)
  (clean shot348)
  (clean shot360)
  (empty shaker391)
  (empty shot118)
  (empty shot348)
  (empty shot360)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker391 l0)
  (shaker-level shaker391 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient237)
  (cocktail-part2 cocktail1 ingredient171)
)
 (:goal
  (and
      (contains shot118 cocktail1)
      (contains shot348 ingredient237)
)))
