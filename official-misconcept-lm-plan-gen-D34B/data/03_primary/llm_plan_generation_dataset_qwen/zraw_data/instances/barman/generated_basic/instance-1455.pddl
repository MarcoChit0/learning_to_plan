(define (problem prob)
 (:domain barman)
 (:objects 
      shaker371 - shaker
      left right - hand
      shot488 shot233 shot299 - shot
      ingredient448 ingredient199 - ingredient
      cocktail257 - cocktail
      dispenser258 dispenser351 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker371)
  (ontable shot488)
  (ontable shot233)
  (ontable shot299)
  (dispenses dispenser258 ingredient448)
  (dispenses dispenser351 ingredient199)
  (clean shaker371)
  (clean shot488)
  (clean shot233)
  (clean shot299)
  (empty shaker371)
  (empty shot488)
  (empty shot233)
  (empty shot299)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker371 l0)
  (shaker-level shaker371 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail257 ingredient199)
  (cocktail-part2 cocktail257 ingredient448)
)
 (:goal
  (and
      (contains shot488 cocktail257)
      (contains shot233 cocktail257)
)))
