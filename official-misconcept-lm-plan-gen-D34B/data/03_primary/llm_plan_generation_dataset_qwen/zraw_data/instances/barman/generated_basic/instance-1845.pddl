(define (problem prob)
 (:domain barman)
 (:objects 
      shaker139 - shaker
      left right - hand
      shot7 shot374 - shot
      ingredient289 ingredient285 ingredient363 - ingredient
      cocktail269 - cocktail
      dispenser461 dispenser83 dispenser188 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker139)
  (ontable shot7)
  (ontable shot374)
  (dispenses dispenser461 ingredient289)
  (dispenses dispenser83 ingredient285)
  (dispenses dispenser188 ingredient363)
  (clean shaker139)
  (clean shot7)
  (clean shot374)
  (empty shaker139)
  (empty shot7)
  (empty shot374)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker139 l0)
  (shaker-level shaker139 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail269 ingredient285)
  (cocktail-part2 cocktail269 ingredient363)
)
 (:goal
  (and
      (contains shot7 cocktail269)
)))
