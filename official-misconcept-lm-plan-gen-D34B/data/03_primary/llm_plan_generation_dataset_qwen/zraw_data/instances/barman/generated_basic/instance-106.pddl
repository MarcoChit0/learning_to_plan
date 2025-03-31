(define (problem prob)
 (:domain barman)
 (:objects 
      shaker151 - shaker
      left right - hand
      shot138 shot363 shot457 - shot
      ingredient12 ingredient57 ingredient250 - ingredient
      cocktail1 - cocktail
      dispenser256 dispenser96 dispenser455 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker151)
  (ontable shot138)
  (ontable shot363)
  (ontable shot457)
  (dispenses dispenser256 ingredient12)
  (dispenses dispenser96 ingredient57)
  (dispenses dispenser455 ingredient250)
  (clean shaker151)
  (clean shot138)
  (clean shot363)
  (clean shot457)
  (empty shaker151)
  (empty shot138)
  (empty shot363)
  (empty shot457)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker151 l0)
  (shaker-level shaker151 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient12)
  (cocktail-part2 cocktail1 ingredient250)
)
 (:goal
  (and
      (contains shot138 cocktail1)
      (contains shot363 cocktail1)
)))
