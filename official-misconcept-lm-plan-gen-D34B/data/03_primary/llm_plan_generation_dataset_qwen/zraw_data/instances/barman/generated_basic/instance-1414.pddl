(define (problem prob)
 (:domain barman)
 (:objects 
      shaker388 - shaker
      left right - hand
      shot498 shot22 - shot
      ingredient315 ingredient373 - ingredient
      cocktail183 - cocktail
      dispenser429 dispenser439 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker388)
  (ontable shot498)
  (ontable shot22)
  (dispenses dispenser429 ingredient315)
  (dispenses dispenser439 ingredient373)
  (clean shaker388)
  (clean shot498)
  (clean shot22)
  (empty shaker388)
  (empty shot498)
  (empty shot22)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker388 l0)
  (shaker-level shaker388 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail183 ingredient315)
  (cocktail-part2 cocktail183 ingredient373)
)
 (:goal
  (and
      (contains shot498 cocktail183)
)))
