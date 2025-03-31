(define (problem prob)
 (:domain barman)
 (:objects 
      shaker241 - shaker
      left right - hand
      shot498 shot403 - shot
      ingredient446 ingredient118 - ingredient
      cocktail1 - cocktail
      dispenser248 dispenser484 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker241)
  (ontable shot498)
  (ontable shot403)
  (dispenses dispenser248 ingredient446)
  (dispenses dispenser484 ingredient118)
  (clean shaker241)
  (clean shot498)
  (clean shot403)
  (empty shaker241)
  (empty shot498)
  (empty shot403)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker241 l0)
  (shaker-level shaker241 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient118)
  (cocktail-part2 cocktail1 ingredient446)
)
 (:goal
  (and
      (contains shot498 cocktail1)
)))
