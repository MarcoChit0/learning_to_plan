(define (problem prob)
 (:domain barman)
 (:objects 
      shaker164 - shaker
      left right - hand
      shot102 shot382 shot15 - shot
      ingredient299 ingredient251 ingredient123 - ingredient
      cocktail265 - cocktail
      dispenser86 dispenser291 dispenser214 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker164)
  (ontable shot102)
  (ontable shot382)
  (ontable shot15)
  (dispenses dispenser86 ingredient299)
  (dispenses dispenser291 ingredient251)
  (dispenses dispenser214 ingredient123)
  (clean shaker164)
  (clean shot102)
  (clean shot382)
  (clean shot15)
  (empty shaker164)
  (empty shot102)
  (empty shot382)
  (empty shot15)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker164 l0)
  (shaker-level shaker164 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail265 ingredient299)
  (cocktail-part2 cocktail265 ingredient123)
)
 (:goal
  (and
      (contains shot102 cocktail265)
      (contains shot382 cocktail265)
)))
