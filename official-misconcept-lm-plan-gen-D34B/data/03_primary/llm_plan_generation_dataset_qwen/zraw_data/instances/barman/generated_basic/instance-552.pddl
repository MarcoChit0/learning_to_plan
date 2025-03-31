(define (problem prob)
 (:domain barman)
 (:objects 
      shaker129 - shaker
      left right - hand
      shot402 - shot
      ingredient498 ingredient483 ingredient89 - ingredient
      cocktail1 - cocktail
      dispenser431 dispenser121 dispenser234 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker129)
  (ontable shot402)
  (dispenses dispenser431 ingredient498)
  (dispenses dispenser121 ingredient483)
  (dispenses dispenser234 ingredient89)
  (clean shaker129)
  (clean shot402)
  (empty shaker129)
  (empty shot402)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker129 l0)
  (shaker-level shaker129 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient498)
  (cocktail-part2 cocktail1 ingredient483)
)
 (:goal
  (and
      (contains shot402 cocktail1)
)))
