(define (problem prob)
 (:domain barman)
 (:objects 
      shaker369 - shaker
      left right - hand
      shot158 shot116 - shot
      ingredient336 ingredient283 ingredient146 - ingredient
      cocktail224 - cocktail
      dispenser28 dispenser316 dispenser144 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker369)
  (ontable shot158)
  (ontable shot116)
  (dispenses dispenser28 ingredient336)
  (dispenses dispenser316 ingredient283)
  (dispenses dispenser144 ingredient146)
  (clean shaker369)
  (clean shot158)
  (clean shot116)
  (empty shaker369)
  (empty shot158)
  (empty shot116)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker369 l0)
  (shaker-level shaker369 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail224 ingredient336)
  (cocktail-part2 cocktail224 ingredient146)
)
 (:goal
  (and
      (contains shot158 cocktail224)
)))
