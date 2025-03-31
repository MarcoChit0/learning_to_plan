(define (problem prob)
 (:domain barman)
 (:objects 
      shaker344 - shaker
      left right - hand
      shot281 shot82 - shot
      ingredient68 ingredient483 ingredient96 ingredient424 - ingredient
      cocktail1 - cocktail
      dispenser139 dispenser321 dispenser377 dispenser256 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker344)
  (ontable shot281)
  (ontable shot82)
  (dispenses dispenser139 ingredient68)
  (dispenses dispenser321 ingredient483)
  (dispenses dispenser377 ingredient96)
  (dispenses dispenser256 ingredient424)
  (clean shaker344)
  (clean shot281)
  (clean shot82)
  (empty shaker344)
  (empty shot281)
  (empty shot82)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker344 l0)
  (shaker-level shaker344 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient68)
  (cocktail-part2 cocktail1 ingredient483)
)
 (:goal
  (and
      (contains shot281 cocktail1)
)))
