(define (problem prob)
 (:domain barman)
 (:objects 
      shaker397 - shaker
      left right - hand
      shot328 shot185 shot279 - shot
      ingredient10 ingredient70 ingredient251 ingredient82 - ingredient
      cocktail239 - cocktail
      dispenser400 dispenser275 dispenser230 dispenser50 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker397)
  (ontable shot328)
  (ontable shot185)
  (ontable shot279)
  (dispenses dispenser400 ingredient10)
  (dispenses dispenser275 ingredient70)
  (dispenses dispenser230 ingredient251)
  (dispenses dispenser50 ingredient82)
  (clean shaker397)
  (clean shot328)
  (clean shot185)
  (clean shot279)
  (empty shaker397)
  (empty shot328)
  (empty shot185)
  (empty shot279)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker397 l0)
  (shaker-level shaker397 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail239 ingredient10)
  (cocktail-part2 cocktail239 ingredient70)
)
 (:goal
  (and
      (contains shot328 cocktail239)
      (contains shot185 cocktail239)
)))
