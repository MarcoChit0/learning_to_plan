(define (problem prob)
 (:domain barman)
 (:objects 
      shaker268 - shaker
      left right - hand
      shot246 shot453 - shot
      ingredient50 ingredient147 ingredient219 ingredient334 - ingredient
      cocktail196 - cocktail
      dispenser127 dispenser207 dispenser305 dispenser244 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker268)
  (ontable shot246)
  (ontable shot453)
  (dispenses dispenser127 ingredient50)
  (dispenses dispenser207 ingredient147)
  (dispenses dispenser305 ingredient219)
  (dispenses dispenser244 ingredient334)
  (clean shaker268)
  (clean shot246)
  (clean shot453)
  (empty shaker268)
  (empty shot246)
  (empty shot453)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker268 l0)
  (shaker-level shaker268 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail196 ingredient219)
  (cocktail-part2 cocktail196 ingredient50)
)
 (:goal
  (and
      (contains shot246 cocktail196)
)))
