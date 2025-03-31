(define (problem prob)
 (:domain barman)
 (:objects 
      shaker23 - shaker
      left right - hand
      shot209 shot80 shot177 - shot
      ingredient36 ingredient465 - ingredient
      cocktail171 - cocktail
      dispenser83 dispenser388 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker23)
  (ontable shot209)
  (ontable shot80)
  (ontable shot177)
  (dispenses dispenser83 ingredient36)
  (dispenses dispenser388 ingredient465)
  (clean shaker23)
  (clean shot209)
  (clean shot80)
  (clean shot177)
  (empty shaker23)
  (empty shot209)
  (empty shot80)
  (empty shot177)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker23 l0)
  (shaker-level shaker23 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail171 ingredient465)
  (cocktail-part2 cocktail171 ingredient36)
)
 (:goal
  (and
      (contains shot209 cocktail171)
      (contains shot80 cocktail171)
)))
