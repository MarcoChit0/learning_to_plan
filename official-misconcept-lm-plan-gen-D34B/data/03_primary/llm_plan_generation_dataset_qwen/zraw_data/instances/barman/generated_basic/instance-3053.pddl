(define (problem prob)
 (:domain barman)
 (:objects 
      shaker216 - shaker
      left right - hand
      shot487 shot38 - shot
      ingredient396 ingredient230 ingredient394 - ingredient
      cocktail365 - cocktail
      dispenser431 dispenser302 dispenser119 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker216)
  (ontable shot487)
  (ontable shot38)
  (dispenses dispenser431 ingredient396)
  (dispenses dispenser302 ingredient230)
  (dispenses dispenser119 ingredient394)
  (clean shaker216)
  (clean shot487)
  (clean shot38)
  (empty shaker216)
  (empty shot487)
  (empty shot38)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker216 l0)
  (shaker-level shaker216 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail365 ingredient230)
  (cocktail-part2 cocktail365 ingredient394)
)
 (:goal
  (and
      (contains shot487 cocktail365)
)))
