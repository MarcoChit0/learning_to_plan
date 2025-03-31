(define (problem prob)
 (:domain barman)
 (:objects 
      shaker215 - shaker
      left right - hand
      shot160 - shot
      ingredient225 ingredient281 ingredient19 ingredient8 - ingredient
      cocktail87 - cocktail
      dispenser151 dispenser45 dispenser449 dispenser404 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker215)
  (ontable shot160)
  (dispenses dispenser151 ingredient225)
  (dispenses dispenser45 ingredient281)
  (dispenses dispenser449 ingredient19)
  (dispenses dispenser404 ingredient8)
  (clean shaker215)
  (clean shot160)
  (empty shaker215)
  (empty shot160)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker215 l0)
  (shaker-level shaker215 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail87 ingredient19)
  (cocktail-part2 cocktail87 ingredient281)
)
 (:goal
  (and
      (contains shot160 cocktail87)
)))
