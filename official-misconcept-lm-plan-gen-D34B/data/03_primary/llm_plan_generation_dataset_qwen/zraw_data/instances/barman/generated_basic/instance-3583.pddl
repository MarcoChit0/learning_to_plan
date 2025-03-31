(define (problem prob)
 (:domain barman)
 (:objects 
      shaker149 - shaker
      left right - hand
      shot77 shot296 - shot
      ingredient475 ingredient140 ingredient339 ingredient225 - ingredient
      cocktail235 - cocktail
      dispenser221 dispenser471 dispenser387 dispenser407 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker149)
  (ontable shot77)
  (ontable shot296)
  (dispenses dispenser221 ingredient475)
  (dispenses dispenser471 ingredient140)
  (dispenses dispenser387 ingredient339)
  (dispenses dispenser407 ingredient225)
  (clean shaker149)
  (clean shot77)
  (clean shot296)
  (empty shaker149)
  (empty shot77)
  (empty shot296)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker149 l0)
  (shaker-level shaker149 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail235 ingredient475)
  (cocktail-part2 cocktail235 ingredient225)
)
 (:goal
  (and
      (contains shot77 cocktail235)
)))
