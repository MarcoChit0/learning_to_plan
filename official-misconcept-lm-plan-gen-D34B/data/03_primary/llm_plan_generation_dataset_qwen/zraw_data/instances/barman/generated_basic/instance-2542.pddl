(define (problem prob)
 (:domain barman)
 (:objects 
      shaker441 - shaker
      left right - hand
      shot24 shot454 - shot
      ingredient239 ingredient22 - ingredient
      cocktail61 - cocktail
      dispenser224 dispenser391 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker441)
  (ontable shot24)
  (ontable shot454)
  (dispenses dispenser224 ingredient239)
  (dispenses dispenser391 ingredient22)
  (clean shaker441)
  (clean shot24)
  (clean shot454)
  (empty shaker441)
  (empty shot24)
  (empty shot454)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker441 l0)
  (shaker-level shaker441 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail61 ingredient22)
  (cocktail-part2 cocktail61 ingredient239)
)
 (:goal
  (and
      (contains shot24 cocktail61)
)))
