(define (problem prob)
 (:domain barman)
 (:objects 
      shaker268 - shaker
      left right - hand
      shot283 shot419 shot414 - shot
      ingredient103 ingredient387 - ingredient
      cocktail473 - cocktail
      dispenser54 dispenser203 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker268)
  (ontable shot283)
  (ontable shot419)
  (ontable shot414)
  (dispenses dispenser54 ingredient103)
  (dispenses dispenser203 ingredient387)
  (clean shaker268)
  (clean shot283)
  (clean shot419)
  (clean shot414)
  (empty shaker268)
  (empty shot283)
  (empty shot419)
  (empty shot414)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker268 l0)
  (shaker-level shaker268 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail473 ingredient103)
  (cocktail-part2 cocktail473 ingredient387)
)
 (:goal
  (and
      (contains shot283 cocktail473)
      (contains shot419 cocktail473)
)))
