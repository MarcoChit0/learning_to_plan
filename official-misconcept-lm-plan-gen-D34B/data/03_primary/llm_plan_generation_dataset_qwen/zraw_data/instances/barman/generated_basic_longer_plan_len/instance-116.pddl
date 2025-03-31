(define (problem prob)
 (:domain barman)
 (:objects 
      shaker374 - shaker
      left right - hand
      shot103 shot82 shot107 - shot
      ingredient404 ingredient483 - ingredient
      cocktail1 cocktail2 - cocktail
      dispenser244 dispenser345 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker374)
  (ontable shot103)
  (ontable shot82)
  (ontable shot107)
  (dispenses dispenser244 ingredient404)
  (dispenses dispenser345 ingredient483)
  (clean shaker374)
  (clean shot103)
  (clean shot82)
  (clean shot107)
  (empty shaker374)
  (empty shot103)
  (empty shot82)
  (empty shot107)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker374 l0)
  (shaker-level shaker374 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient404)
  (cocktail-part2 cocktail1 ingredient483)
  (cocktail-part1 cocktail2 ingredient404)
  (cocktail-part2 cocktail2 ingredient483)
)
 (:goal
  (and
      (contains shot103 cocktail1)
      (contains shot82 cocktail2)
)))
