(define (problem prob)
 (:domain barman)
 (:objects 
      shaker170 - shaker
      left right - hand
      shot331 - shot
      ingredient52 ingredient291 ingredient165 ingredient439 - ingredient
      cocktail1 - cocktail
      dispenser173 dispenser396 dispenser386 dispenser331 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker170)
  (ontable shot331)
  (dispenses dispenser173 ingredient52)
  (dispenses dispenser396 ingredient291)
  (dispenses dispenser386 ingredient165)
  (dispenses dispenser331 ingredient439)
  (clean shaker170)
  (clean shot331)
  (empty shaker170)
  (empty shot331)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker170 l0)
  (shaker-level shaker170 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient52)
  (cocktail-part2 cocktail1 ingredient291)
)
 (:goal
  (and
      (contains shot331 cocktail1)
)))
