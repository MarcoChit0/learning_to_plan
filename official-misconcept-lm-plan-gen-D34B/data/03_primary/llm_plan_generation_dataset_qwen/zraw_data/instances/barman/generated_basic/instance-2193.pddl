(define (problem prob)
 (:domain barman)
 (:objects 
      shaker92 - shaker
      left right - hand
      shot177 shot72 shot172 - shot
      ingredient483 ingredient128 - ingredient
      cocktail332 - cocktail
      dispenser230 dispenser84 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker92)
  (ontable shot177)
  (ontable shot72)
  (ontable shot172)
  (dispenses dispenser230 ingredient483)
  (dispenses dispenser84 ingredient128)
  (clean shaker92)
  (clean shot177)
  (clean shot72)
  (clean shot172)
  (empty shaker92)
  (empty shot177)
  (empty shot72)
  (empty shot172)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker92 l0)
  (shaker-level shaker92 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail332 ingredient128)
  (cocktail-part2 cocktail332 ingredient483)
)
 (:goal
  (and
      (contains shot177 cocktail332)
      (contains shot72 cocktail332)
)))
