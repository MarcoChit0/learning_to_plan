(define (problem prob)
 (:domain barman)
 (:objects 
      shaker327 - shaker
      left right - hand
      shot498 shot66 - shot
      ingredient426 ingredient333 - ingredient
      cocktail1 - cocktail
      dispenser221 dispenser279 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker327)
  (ontable shot498)
  (ontable shot66)
  (dispenses dispenser221 ingredient426)
  (dispenses dispenser279 ingredient333)
  (clean shaker327)
  (clean shot498)
  (clean shot66)
  (empty shaker327)
  (empty shot498)
  (empty shot66)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker327 l0)
  (shaker-level shaker327 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient333)
  (cocktail-part2 cocktail1 ingredient426)
)
 (:goal
  (and
      (contains shot498 cocktail1)
)))
