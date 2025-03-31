(define (problem prob)
 (:domain barman)
 (:objects 
      shaker90 - shaker
      left right - hand
      shot444 shot302 - shot
      ingredient396 ingredient201 - ingredient
      cocktail281 - cocktail
      dispenser294 dispenser333 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker90)
  (ontable shot444)
  (ontable shot302)
  (dispenses dispenser294 ingredient396)
  (dispenses dispenser333 ingredient201)
  (clean shaker90)
  (clean shot444)
  (clean shot302)
  (empty shaker90)
  (empty shot444)
  (empty shot302)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker90 l0)
  (shaker-level shaker90 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail281 ingredient396)
  (cocktail-part2 cocktail281 ingredient201)
)
 (:goal
  (and
      (contains shot444 cocktail281)
)))
