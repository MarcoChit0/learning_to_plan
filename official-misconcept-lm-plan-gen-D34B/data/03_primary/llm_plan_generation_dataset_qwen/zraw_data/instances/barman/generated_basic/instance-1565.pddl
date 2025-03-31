(define (problem prob)
 (:domain barman)
 (:objects 
      shaker213 - shaker
      left right - hand
      shot458 shot348 shot189 - shot
      ingredient277 ingredient26 ingredient135 ingredient416 - ingredient
      cocktail473 - cocktail
      dispenser248 dispenser312 dispenser385 dispenser216 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker213)
  (ontable shot458)
  (ontable shot348)
  (ontable shot189)
  (dispenses dispenser248 ingredient277)
  (dispenses dispenser312 ingredient26)
  (dispenses dispenser385 ingredient135)
  (dispenses dispenser216 ingredient416)
  (clean shaker213)
  (clean shot458)
  (clean shot348)
  (clean shot189)
  (empty shaker213)
  (empty shot458)
  (empty shot348)
  (empty shot189)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker213 l0)
  (shaker-level shaker213 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail473 ingredient277)
  (cocktail-part2 cocktail473 ingredient416)
)
 (:goal
  (and
      (contains shot458 cocktail473)
      (contains shot348 cocktail473)
)))
