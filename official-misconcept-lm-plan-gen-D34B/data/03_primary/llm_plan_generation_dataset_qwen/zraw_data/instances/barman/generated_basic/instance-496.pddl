(define (problem prob)
 (:domain barman)
 (:objects 
      shaker15 - shaker
      left right - hand
      shot317 shot120 - shot
      ingredient101 ingredient176 ingredient123 ingredient456 - ingredient
      cocktail1 - cocktail
      dispenser129 dispenser92 dispenser1 dispenser368 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker15)
  (ontable shot317)
  (ontable shot120)
  (dispenses dispenser129 ingredient101)
  (dispenses dispenser92 ingredient176)
  (dispenses dispenser1 ingredient123)
  (dispenses dispenser368 ingredient456)
  (clean shaker15)
  (clean shot317)
  (clean shot120)
  (empty shaker15)
  (empty shot317)
  (empty shot120)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker15 l0)
  (shaker-level shaker15 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient176)
  (cocktail-part2 cocktail1 ingredient123)
)
 (:goal
  (and
      (contains shot317 cocktail1)
)))
