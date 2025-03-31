(define (problem prob)
 (:domain barman)
 (:objects 
      shaker419 - shaker
      left right - hand
      shot475 - shot
      ingredient270 ingredient498 ingredient79 - ingredient
      cocktail158 - cocktail
      dispenser161 dispenser207 dispenser188 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker419)
  (ontable shot475)
  (dispenses dispenser161 ingredient270)
  (dispenses dispenser207 ingredient498)
  (dispenses dispenser188 ingredient79)
  (clean shaker419)
  (clean shot475)
  (empty shaker419)
  (empty shot475)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker419 l0)
  (shaker-level shaker419 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail158 ingredient270)
  (cocktail-part2 cocktail158 ingredient79)
)
 (:goal
  (and
      (contains shot475 cocktail158)
)))
