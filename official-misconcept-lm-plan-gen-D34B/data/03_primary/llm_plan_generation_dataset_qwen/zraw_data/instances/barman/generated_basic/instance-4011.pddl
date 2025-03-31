(define (problem prob)
 (:domain barman)
 (:objects 
      shaker401 - shaker
      left right - hand
      shot72 shot258 shot179 - shot
      ingredient255 ingredient248 - ingredient
      cocktail310 - cocktail
      dispenser282 dispenser376 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker401)
  (ontable shot72)
  (ontable shot258)
  (ontable shot179)
  (dispenses dispenser282 ingredient255)
  (dispenses dispenser376 ingredient248)
  (clean shaker401)
  (clean shot72)
  (clean shot258)
  (clean shot179)
  (empty shaker401)
  (empty shot72)
  (empty shot258)
  (empty shot179)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker401 l0)
  (shaker-level shaker401 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail310 ingredient248)
  (cocktail-part2 cocktail310 ingredient255)
)
 (:goal
  (and
      (contains shot72 cocktail310)
      (contains shot258 ingredient255)
)))
