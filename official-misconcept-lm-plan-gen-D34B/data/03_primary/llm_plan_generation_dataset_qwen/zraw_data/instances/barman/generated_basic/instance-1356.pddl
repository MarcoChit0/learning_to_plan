(define (problem prob)
 (:domain barman)
 (:objects 
      shaker147 - shaker
      left right - hand
      shot392 - shot
      ingredient316 ingredient281 ingredient77 - ingredient
      cocktail84 - cocktail
      dispenser31 dispenser71 dispenser211 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker147)
  (ontable shot392)
  (dispenses dispenser31 ingredient316)
  (dispenses dispenser71 ingredient281)
  (dispenses dispenser211 ingredient77)
  (clean shaker147)
  (clean shot392)
  (empty shaker147)
  (empty shot392)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker147 l0)
  (shaker-level shaker147 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail84 ingredient316)
  (cocktail-part2 cocktail84 ingredient281)
)
 (:goal
  (and
      (contains shot392 cocktail84)
)))
