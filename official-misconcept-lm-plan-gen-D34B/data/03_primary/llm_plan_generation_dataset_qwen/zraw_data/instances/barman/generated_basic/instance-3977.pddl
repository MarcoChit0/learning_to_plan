(define (problem prob)
 (:domain barman)
 (:objects 
      shaker498 - shaker
      left right - hand
      shot477 shot142 - shot
      ingredient122 ingredient81 ingredient146 ingredient160 - ingredient
      cocktail64 - cocktail
      dispenser296 dispenser41 dispenser242 dispenser262 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker498)
  (ontable shot477)
  (ontable shot142)
  (dispenses dispenser296 ingredient122)
  (dispenses dispenser41 ingredient81)
  (dispenses dispenser242 ingredient146)
  (dispenses dispenser262 ingredient160)
  (clean shaker498)
  (clean shot477)
  (clean shot142)
  (empty shaker498)
  (empty shot477)
  (empty shot142)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker498 l0)
  (shaker-level shaker498 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail64 ingredient122)
  (cocktail-part2 cocktail64 ingredient146)
)
 (:goal
  (and
      (contains shot477 cocktail64)
)))
