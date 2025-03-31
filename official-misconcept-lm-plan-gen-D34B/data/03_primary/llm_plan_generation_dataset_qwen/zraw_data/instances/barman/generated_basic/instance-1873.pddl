(define (problem prob)
 (:domain barman)
 (:objects 
      shaker315 - shaker
      left right - hand
      shot49 shot138 - shot
      ingredient172 ingredient176 - ingredient
      cocktail25 - cocktail
      dispenser416 dispenser443 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker315)
  (ontable shot49)
  (ontable shot138)
  (dispenses dispenser416 ingredient172)
  (dispenses dispenser443 ingredient176)
  (clean shaker315)
  (clean shot49)
  (clean shot138)
  (empty shaker315)
  (empty shot49)
  (empty shot138)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker315 l0)
  (shaker-level shaker315 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail25 ingredient176)
  (cocktail-part2 cocktail25 ingredient172)
)
 (:goal
  (and
      (contains shot49 cocktail25)
)))
