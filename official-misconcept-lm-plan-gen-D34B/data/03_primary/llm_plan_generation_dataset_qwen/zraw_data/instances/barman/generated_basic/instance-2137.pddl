(define (problem prob)
 (:domain barman)
 (:objects 
      shaker226 - shaker
      left right - hand
      shot177 shot475 shot7 - shot
      ingredient319 ingredient220 - ingredient
      cocktail340 - cocktail
      dispenser446 dispenser417 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker226)
  (ontable shot177)
  (ontable shot475)
  (ontable shot7)
  (dispenses dispenser446 ingredient319)
  (dispenses dispenser417 ingredient220)
  (clean shaker226)
  (clean shot177)
  (clean shot475)
  (clean shot7)
  (empty shaker226)
  (empty shot177)
  (empty shot475)
  (empty shot7)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker226 l0)
  (shaker-level shaker226 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail340 ingredient220)
  (cocktail-part2 cocktail340 ingredient319)
)
 (:goal
  (and
      (contains shot177 cocktail340)
      (contains shot475 ingredient319)
)))
