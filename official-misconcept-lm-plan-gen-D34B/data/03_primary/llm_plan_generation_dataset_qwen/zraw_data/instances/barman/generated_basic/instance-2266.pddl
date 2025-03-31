(define (problem prob)
 (:domain barman)
 (:objects 
      shaker457 - shaker
      left right - hand
      shot253 shot7 - shot
      ingredient423 ingredient464 ingredient124 ingredient255 - ingredient
      cocktail240 - cocktail
      dispenser496 dispenser464 dispenser149 dispenser348 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker457)
  (ontable shot253)
  (ontable shot7)
  (dispenses dispenser496 ingredient423)
  (dispenses dispenser464 ingredient464)
  (dispenses dispenser149 ingredient124)
  (dispenses dispenser348 ingredient255)
  (clean shaker457)
  (clean shot253)
  (clean shot7)
  (empty shaker457)
  (empty shot253)
  (empty shot7)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker457 l0)
  (shaker-level shaker457 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail240 ingredient464)
  (cocktail-part2 cocktail240 ingredient255)
)
 (:goal
  (and
      (contains shot253 cocktail240)
)))
