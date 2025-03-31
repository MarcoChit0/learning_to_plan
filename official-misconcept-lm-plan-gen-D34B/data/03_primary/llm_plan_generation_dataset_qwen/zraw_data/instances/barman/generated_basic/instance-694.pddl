(define (problem prob)
 (:domain barman)
 (:objects 
      shaker305 - shaker
      left right - hand
      shot300 shot159 - shot
      ingredient163 ingredient75 ingredient335 - ingredient
      cocktail1 - cocktail
      dispenser356 dispenser39 dispenser376 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker305)
  (ontable shot300)
  (ontable shot159)
  (dispenses dispenser356 ingredient163)
  (dispenses dispenser39 ingredient75)
  (dispenses dispenser376 ingredient335)
  (clean shaker305)
  (clean shot300)
  (clean shot159)
  (empty shaker305)
  (empty shot300)
  (empty shot159)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker305 l0)
  (shaker-level shaker305 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient163)
  (cocktail-part2 cocktail1 ingredient335)
)
 (:goal
  (and
      (contains shot300 cocktail1)
)))
