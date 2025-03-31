(define (problem prob)
 (:domain barman)
 (:objects 
      shaker457 - shaker
      left right - hand
      shot453 shot4 - shot
      ingredient47 ingredient403 ingredient89 ingredient351 - ingredient
      cocktail104 - cocktail
      dispenser365 dispenser151 dispenser52 dispenser394 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker457)
  (ontable shot453)
  (ontable shot4)
  (dispenses dispenser365 ingredient47)
  (dispenses dispenser151 ingredient403)
  (dispenses dispenser52 ingredient89)
  (dispenses dispenser394 ingredient351)
  (clean shaker457)
  (clean shot453)
  (clean shot4)
  (empty shaker457)
  (empty shot453)
  (empty shot4)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker457 l0)
  (shaker-level shaker457 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail104 ingredient351)
  (cocktail-part2 cocktail104 ingredient89)
)
 (:goal
  (and
      (contains shot453 cocktail104)
)))
