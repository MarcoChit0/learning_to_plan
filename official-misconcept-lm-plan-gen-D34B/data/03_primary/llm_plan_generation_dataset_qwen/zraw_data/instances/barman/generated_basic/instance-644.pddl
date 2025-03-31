(define (problem prob)
 (:domain barman)
 (:objects 
      shaker477 - shaker
      left right - hand
      shot437 shot4 shot69 - shot
      ingredient463 ingredient459 ingredient292 - ingredient
      cocktail1 - cocktail
      dispenser230 dispenser134 dispenser3 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker477)
  (ontable shot437)
  (ontable shot4)
  (ontable shot69)
  (dispenses dispenser230 ingredient463)
  (dispenses dispenser134 ingredient459)
  (dispenses dispenser3 ingredient292)
  (clean shaker477)
  (clean shot437)
  (clean shot4)
  (clean shot69)
  (empty shaker477)
  (empty shot437)
  (empty shot4)
  (empty shot69)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker477 l0)
  (shaker-level shaker477 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient463)
  (cocktail-part2 cocktail1 ingredient292)
)
 (:goal
  (and
      (contains shot437 cocktail1)
      (contains shot4 cocktail1)
)))
