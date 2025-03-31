(define (problem prob)
 (:domain barman)
 (:objects 
      shaker170 - shaker
      left right - hand
      shot447 shot71 - shot
      ingredient241 ingredient367 - ingredient
      cocktail1 - cocktail
      dispenser311 dispenser123 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker170)
  (ontable shot447)
  (ontable shot71)
  (dispenses dispenser311 ingredient241)
  (dispenses dispenser123 ingredient367)
  (clean shaker170)
  (clean shot447)
  (clean shot71)
  (empty shaker170)
  (empty shot447)
  (empty shot71)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker170 l0)
  (shaker-level shaker170 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient367)
  (cocktail-part2 cocktail1 ingredient241)
)
 (:goal
  (and
      (contains shot447 cocktail1)
)))
