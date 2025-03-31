(define (problem prob)
 (:domain barman)
 (:objects 
      shaker78 - shaker
      left right - hand
      shot89 shot372 - shot
      ingredient50 ingredient129 ingredient107 ingredient392 - ingredient
      cocktail155 - cocktail
      dispenser268 dispenser82 dispenser469 dispenser229 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker78)
  (ontable shot89)
  (ontable shot372)
  (dispenses dispenser268 ingredient50)
  (dispenses dispenser82 ingredient129)
  (dispenses dispenser469 ingredient107)
  (dispenses dispenser229 ingredient392)
  (clean shaker78)
  (clean shot89)
  (clean shot372)
  (empty shaker78)
  (empty shot89)
  (empty shot372)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker78 l0)
  (shaker-level shaker78 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail155 ingredient50)
  (cocktail-part2 cocktail155 ingredient107)
)
 (:goal
  (and
      (contains shot89 cocktail155)
)))
