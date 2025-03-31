(define (problem prob)
 (:domain barman)
 (:objects 
      shaker209 - shaker
      left right - hand
      shot434 shot206 shot363 - shot
      ingredient363 ingredient400 ingredient34 ingredient204 - ingredient
      cocktail114 - cocktail
      dispenser484 dispenser38 dispenser242 dispenser392 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker209)
  (ontable shot434)
  (ontable shot206)
  (ontable shot363)
  (dispenses dispenser484 ingredient363)
  (dispenses dispenser38 ingredient400)
  (dispenses dispenser242 ingredient34)
  (dispenses dispenser392 ingredient204)
  (clean shaker209)
  (clean shot434)
  (clean shot206)
  (clean shot363)
  (empty shaker209)
  (empty shot434)
  (empty shot206)
  (empty shot363)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker209 l0)
  (shaker-level shaker209 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail114 ingredient34)
  (cocktail-part2 cocktail114 ingredient363)
)
 (:goal
  (and
      (contains shot434 cocktail114)
      (contains shot206 ingredient400)
)))
