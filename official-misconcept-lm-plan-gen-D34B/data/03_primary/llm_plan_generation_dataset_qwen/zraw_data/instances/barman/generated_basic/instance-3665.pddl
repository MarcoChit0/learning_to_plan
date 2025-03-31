(define (problem prob)
 (:domain barman)
 (:objects 
      shaker31 - shaker
      left right - hand
      shot484 shot89 - shot
      ingredient229 ingredient52 ingredient168 ingredient12 - ingredient
      cocktail293 - cocktail
      dispenser459 dispenser230 dispenser20 dispenser404 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker31)
  (ontable shot484)
  (ontable shot89)
  (dispenses dispenser459 ingredient229)
  (dispenses dispenser230 ingredient52)
  (dispenses dispenser20 ingredient168)
  (dispenses dispenser404 ingredient12)
  (clean shaker31)
  (clean shot484)
  (clean shot89)
  (empty shaker31)
  (empty shot484)
  (empty shot89)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker31 l0)
  (shaker-level shaker31 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail293 ingredient52)
  (cocktail-part2 cocktail293 ingredient12)
)
 (:goal
  (and
      (contains shot484 cocktail293)
)))
