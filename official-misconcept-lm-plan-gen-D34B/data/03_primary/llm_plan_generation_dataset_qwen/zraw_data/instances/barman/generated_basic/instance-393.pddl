(define (problem prob)
 (:domain barman)
 (:objects 
      shaker67 - shaker
      left right - hand
      shot127 shot96 shot160 - shot
      ingredient255 ingredient280 ingredient259 - ingredient
      cocktail1 - cocktail
      dispenser451 dispenser460 dispenser179 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker67)
  (ontable shot127)
  (ontable shot96)
  (ontable shot160)
  (dispenses dispenser451 ingredient255)
  (dispenses dispenser460 ingredient280)
  (dispenses dispenser179 ingredient259)
  (clean shaker67)
  (clean shot127)
  (clean shot96)
  (clean shot160)
  (empty shaker67)
  (empty shot127)
  (empty shot96)
  (empty shot160)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker67 l0)
  (shaker-level shaker67 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient259)
  (cocktail-part2 cocktail1 ingredient280)
)
 (:goal
  (and
      (contains shot127 cocktail1)
      (contains shot96 cocktail1)
)))
