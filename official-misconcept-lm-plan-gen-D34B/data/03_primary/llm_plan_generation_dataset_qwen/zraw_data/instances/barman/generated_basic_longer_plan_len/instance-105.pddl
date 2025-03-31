(define (problem prob)
 (:domain barman)
 (:objects 
      shaker478 - shaker
      left right - hand
      shot14 shot157 shot412 - shot
      ingredient349 ingredient61 ingredient470 ingredient5 - ingredient
      cocktail1 - cocktail
      dispenser250 dispenser412 dispenser221 dispenser100 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker478)
  (ontable shot14)
  (ontable shot157)
  (ontable shot412)
  (dispenses dispenser250 ingredient349)
  (dispenses dispenser412 ingredient61)
  (dispenses dispenser221 ingredient470)
  (dispenses dispenser100 ingredient5)
  (clean shaker478)
  (clean shot14)
  (clean shot157)
  (clean shot412)
  (empty shaker478)
  (empty shot14)
  (empty shot157)
  (empty shot412)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker478 l0)
  (shaker-level shaker478 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient349)
  (cocktail-part2 cocktail1 ingredient5)
)
 (:goal
  (and
      (contains shot14 cocktail1)
      (contains shot157 ingredient349)
)))
