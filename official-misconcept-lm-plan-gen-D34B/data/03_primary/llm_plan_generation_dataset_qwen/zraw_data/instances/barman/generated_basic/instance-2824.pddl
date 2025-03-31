(define (problem prob)
 (:domain barman)
 (:objects 
      shaker29 - shaker
      left right - hand
      shot397 shot246 shot418 - shot
      ingredient55 ingredient292 ingredient174 ingredient393 - ingredient
      cocktail234 - cocktail
      dispenser318 dispenser496 dispenser141 dispenser273 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker29)
  (ontable shot397)
  (ontable shot246)
  (ontable shot418)
  (dispenses dispenser318 ingredient55)
  (dispenses dispenser496 ingredient292)
  (dispenses dispenser141 ingredient174)
  (dispenses dispenser273 ingredient393)
  (clean shaker29)
  (clean shot397)
  (clean shot246)
  (clean shot418)
  (empty shaker29)
  (empty shot397)
  (empty shot246)
  (empty shot418)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker29 l0)
  (shaker-level shaker29 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail234 ingredient292)
  (cocktail-part2 cocktail234 ingredient55)
)
 (:goal
  (and
      (contains shot397 cocktail234)
      (contains shot246 ingredient174)
)))
