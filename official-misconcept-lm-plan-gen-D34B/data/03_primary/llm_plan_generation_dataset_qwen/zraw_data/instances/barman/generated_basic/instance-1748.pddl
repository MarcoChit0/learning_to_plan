(define (problem prob)
 (:domain barman)
 (:objects 
      shaker443 - shaker
      left right - hand
      shot76 shot270 - shot
      ingredient108 ingredient231 ingredient24 - ingredient
      cocktail353 - cocktail
      dispenser71 dispenser447 dispenser254 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker443)
  (ontable shot76)
  (ontable shot270)
  (dispenses dispenser71 ingredient108)
  (dispenses dispenser447 ingredient231)
  (dispenses dispenser254 ingredient24)
  (clean shaker443)
  (clean shot76)
  (clean shot270)
  (empty shaker443)
  (empty shot76)
  (empty shot270)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker443 l0)
  (shaker-level shaker443 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail353 ingredient231)
  (cocktail-part2 cocktail353 ingredient108)
)
 (:goal
  (and
      (contains shot76 cocktail353)
)))
