(define (problem prob)
 (:domain barman)
 (:objects 
      shaker399 - shaker
      left right - hand
      shot88 shot355 shot201 - shot
      ingredient126 ingredient287 - ingredient
      cocktail1 - cocktail
      dispenser418 dispenser361 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker399)
  (ontable shot88)
  (ontable shot355)
  (ontable shot201)
  (dispenses dispenser418 ingredient126)
  (dispenses dispenser361 ingredient287)
  (clean shaker399)
  (clean shot88)
  (clean shot355)
  (clean shot201)
  (empty shaker399)
  (empty shot88)
  (empty shot355)
  (empty shot201)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker399 l0)
  (shaker-level shaker399 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient287)
  (cocktail-part2 cocktail1 ingredient126)
)
 (:goal
  (and
      (contains shot88 cocktail1)
      (contains shot355 ingredient126)
)))
