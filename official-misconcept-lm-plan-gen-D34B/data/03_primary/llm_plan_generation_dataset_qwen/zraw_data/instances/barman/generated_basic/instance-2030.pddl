(define (problem prob)
 (:domain barman)
 (:objects 
      shaker371 - shaker
      left right - hand
      shot414 shot313 - shot
      ingredient139 ingredient129 - ingredient
      cocktail269 - cocktail
      dispenser149 dispenser79 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker371)
  (ontable shot414)
  (ontable shot313)
  (dispenses dispenser149 ingredient139)
  (dispenses dispenser79 ingredient129)
  (clean shaker371)
  (clean shot414)
  (clean shot313)
  (empty shaker371)
  (empty shot414)
  (empty shot313)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker371 l0)
  (shaker-level shaker371 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail269 ingredient139)
  (cocktail-part2 cocktail269 ingredient129)
)
 (:goal
  (and
      (contains shot414 cocktail269)
)))
