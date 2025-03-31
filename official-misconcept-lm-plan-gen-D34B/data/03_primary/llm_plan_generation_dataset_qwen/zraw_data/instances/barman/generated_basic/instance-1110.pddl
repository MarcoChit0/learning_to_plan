(define (problem prob)
 (:domain barman)
 (:objects 
      shaker372 - shaker
      left right - hand
      shot260 shot152 - shot
      ingredient477 ingredient382 - ingredient
      cocktail1 - cocktail
      dispenser135 dispenser151 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker372)
  (ontable shot260)
  (ontable shot152)
  (dispenses dispenser135 ingredient477)
  (dispenses dispenser151 ingredient382)
  (clean shaker372)
  (clean shot260)
  (clean shot152)
  (empty shaker372)
  (empty shot260)
  (empty shot152)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker372 l0)
  (shaker-level shaker372 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient382)
  (cocktail-part2 cocktail1 ingredient477)
)
 (:goal
  (and
      (contains shot260 cocktail1)
)))
