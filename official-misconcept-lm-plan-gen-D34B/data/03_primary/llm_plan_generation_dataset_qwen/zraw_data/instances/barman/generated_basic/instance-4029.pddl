(define (problem prob)
 (:domain barman)
 (:objects 
      shaker22 - shaker
      left right - hand
      shot237 shot170 - shot
      ingredient138 ingredient465 ingredient341 - ingredient
      cocktail443 - cocktail
      dispenser165 dispenser226 dispenser150 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker22)
  (ontable shot237)
  (ontable shot170)
  (dispenses dispenser165 ingredient138)
  (dispenses dispenser226 ingredient465)
  (dispenses dispenser150 ingredient341)
  (clean shaker22)
  (clean shot237)
  (clean shot170)
  (empty shaker22)
  (empty shot237)
  (empty shot170)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker22 l0)
  (shaker-level shaker22 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail443 ingredient341)
  (cocktail-part2 cocktail443 ingredient465)
)
 (:goal
  (and
      (contains shot237 cocktail443)
)))
