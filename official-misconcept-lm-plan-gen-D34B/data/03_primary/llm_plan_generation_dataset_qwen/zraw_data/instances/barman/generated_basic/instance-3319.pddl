(define (problem prob)
 (:domain barman)
 (:objects 
      shaker229 - shaker
      left right - hand
      shot426 shot86 shot361 - shot
      ingredient174 ingredient91 ingredient240 - ingredient
      cocktail462 - cocktail
      dispenser146 dispenser189 dispenser413 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker229)
  (ontable shot426)
  (ontable shot86)
  (ontable shot361)
  (dispenses dispenser146 ingredient174)
  (dispenses dispenser189 ingredient91)
  (dispenses dispenser413 ingredient240)
  (clean shaker229)
  (clean shot426)
  (clean shot86)
  (clean shot361)
  (empty shaker229)
  (empty shot426)
  (empty shot86)
  (empty shot361)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker229 l0)
  (shaker-level shaker229 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail462 ingredient240)
  (cocktail-part2 cocktail462 ingredient91)
)
 (:goal
  (and
      (contains shot426 cocktail462)
      (contains shot86 cocktail462)
)))
